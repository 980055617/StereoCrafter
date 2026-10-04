"""CONTROL B step 1 -- register the REAL right-eye GT to the render frame, per clip.

For each clip the train tile video_data/train/<clip>_train.mp4 is a 2x2 tile of half-size quadrants
    TL = real left eye   TR = real right eye (the LPIPS target)   BL = occlusion mask   BR = warped right eye
BR is the model's own input, so it is pixel-registered with every render by construction.  We look for the
integer window shift (ddy, ddx), |ddy| <= 3, |ddx| <= 64, that maximises the PSNR between
    TR[top+ddy : top+ddy+576, left+ddx : left+ddx+1024]   and   BR[top : top+576, left : left+1024]
outside the disocclusion holes (BL > 0.5), where (top, left) is the deployed / trainer / scorer crop window
(utils/inpainting.py:147-158 + inpainting_inference.py:258-262, == xcheck_mini_ft.py:crop_quadrants).
The search is EXHAUSTIVE on the full 576x1024 window (no coarse-to-fine), for frames spaced 10 apart.
The clip-global shift is the (ddy, ddx) that maximises the MEAN PSNR over the sampled frames; the per-frame
optima and a per-block (3x4 grid) fit at three frames are reported as the stability / spatial-variation check.

The registered GT of ALL frames is then written LOSSLESSLY (FFV1, the beyond4 writer) to
    <clip>_gt_registered_576x1024.mkv   (+ .md5 of the pre-encode array; decode is verified bit-exact)
Columns/rows the shifted window would expose outside the quadrant are filled from BR (counted; 0 expected).
Nothing tracked is modified; nothing under video_data/ is written.
usage: CUDA_VISIBLE_DEVICES=1 python register_gt_regB.py 0301 0204
"""
import os, sys, json, hashlib, math, time
import numpy as np
import torch
import cv2
from decord import VideoReader, cpu

REPO = "/home/kawa/master_project/StereoCrafter"; os.chdir(REPO); sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts/distill/runs/fulldata_v2/beyond4"))
import infer_lossless as IL   # reuse _ffv1_write (FFV1 level 3, bgr0, -g 1); importing also rebinds II.write_video_opencv, unused here

HERE = os.path.dirname(os.path.abspath(__file__))
TH, TW = 576, 1024
DDY, DDX = 3, 64
FRAME_STEP = 10
BLOCK_FRAMES = (0, 75, 150)

def crop_window(H, W):
    """(top, left) of the deployed 576x1024 window inside a quadrant (crop to /128 then centre), as the trainer/pipeline."""
    h, w = H // 128 * 128, W // 128 * 128
    return (h - TH) // 2, (w - TW) // 2

def psnr_from_mse(m):
    return 10 * math.log10(255.0 ** 2 / max(m, 1e-12))

def masked_psnr_grid(TRq, BRw, valid, top, left, ddys, ddxs):
    """PSNR[ddy, ddx] of TR shifted-window vs the BR window over valid pixels (all 3 channels). TRq float32 [H,W,3]."""
    out = np.full((len(ddys), len(ddxs)), np.nan)
    v = valid.float().unsqueeze(-1)
    nval = float(v.sum().item()) * 3.0
    for i, ddy in enumerate(ddys):
        for j, ddx in enumerate(ddxs):
            t, l = top + ddy, left + ddx
            if t < 0 or l < 0 or t + TH > TRq.shape[0] or l + TW > TRq.shape[1]:
                continue
            d = TRq[t:t + TH, l:l + TW] - BRw
            out[i, j] = psnr_from_mse(float((d * d * v).sum().item()) / nval)
    return out

def run_clip(clip):
    t_all = time.time()
    path = f"video_data/train/{clip}_train.mp4"
    vr = VideoReader(path, ctx=cpu(0)); n = len(vr); fps = float(vr.get_avg_fps())
    f0 = vr[0].asnumpy(); H, W = f0.shape[0] // 2, f0.shape[1] // 2
    top, left = crop_window(H, W)
    # the scorer's window for this clip (score_clip_ll.py: t0=(H-h)//2+dy, l0=(W-w)//2+dx with (dy,dx)=(-28,0) on 0301/0204)
    sc_t0, sc_l0 = (H - TH) // 2 - 28, (W - TW) // 2 + 0
    assert (top, left) == (sc_t0, sc_l0), f"{clip}: trainer window {(top, left)} != scorer window {(sc_t0, sc_l0)}"
    print(f"[{clip}] n={n} fps={fps:.3f} quadrant {H}x{W} window (top,left)=({top},{left}) == scorer window", flush=True)
    ddys = list(range(-DDY, DDY + 1)); ddxs = list(range(-DDX, DDX + 1))

    # --- mask cross-check: train tile BL vs splatting video BL (the pipeline's mask) at frame 0 ---
    sp = VideoReader(f"video_data/splatting/{clip}_splatting_results.mp4", ctx=cpu(0))
    s0 = sp[0].asnumpy()
    m_train = f0[H:, :W].astype(np.float32).mean(-1)[top:top + TH, left:left + TW] > 127.5
    m_splat = s0[H:, :W].astype(np.float32).mean(-1)[top:top + TH, left:left + TW] > 127.5
    print(f"[{clip}] mask frame0: train-tile BL hole frac {m_train.mean()*100:.3f}% | splatting BL hole frac {m_splat.mean()*100:.3f}% | disagree {(m_train != m_splat).mean()*100:.4f}% of pixels", flush=True)

    # --- per-frame exhaustive search ---
    frames = list(range(0, n, FRAME_STEP))
    if frames[-1] != n - 1: frames.append(n - 1)
    per_frame = []; grids = []
    for fi in frames:
        f = vr[fi].asnumpy()
        TR = torch.from_numpy(f[:H, W:2 * W].astype(np.float32))
        BR = torch.from_numpy(f[H:, W:2 * W].astype(np.float32))[top:top + TH, left:left + TW]
        hole = torch.from_numpy(f[H:, :W].astype(np.float32).mean(-1)[top:top + TH, left:left + TW] > 127.5)
        valid = ~hole
        g = masked_psnr_grid(TR, BR, valid, top, left, ddys, ddxs)
        bi, bj = np.unravel_index(np.nanargmax(g), g.shape)
        p0 = g[ddys.index(0), ddxs.index(0)]
        # peakedness along ddx at the best ddy
        row = g[bi]
        within1db = [ddxs[j] for j in range(len(ddxs)) if row[j] >= row[bj] - 1.0]
        per_frame.append(dict(frame=fi, ddy=ddys[bi], ddx=ddxs[bj], psnr_best=float(g[bi, bj]), psnr_zero=float(p0),
                              hole_frac=float(hole.float().mean()), ddx_within_1dB=[min(within1db), max(within1db)]))
        grids.append(g)
        print(f"[{clip}] frame {fi:3d}: best (ddy,ddx)=({ddys[bi]:+d},{ddxs[bj]:+d}) PSNR {g[bi,bj]:.2f} dB (at (0,0): {p0:.2f} dB, gain {g[bi,bj]-p0:+.2f}) hole {hole.float().mean()*100:.2f}%  ddx within 1 dB of peak: [{min(within1db)},{max(within1db)}]", flush=True)
    G = np.stack(grids)                      # [F, 7, 129]
    mean_g = np.nanmean(G, axis=0)
    gi, gj = np.unravel_index(np.nanargmax(mean_g), mean_g.shape)
    g_ddy, g_ddx = ddys[gi], ddxs[gj]
    ddx_list = [p["ddx"] for p in per_frame]; ddy_list = [p["ddy"] for p in per_frame]
    med = (int(np.median(ddy_list)), int(np.median(ddx_list)))
    spread = max(ddx_list) - min(ddx_list)
    print(f"[{clip}] JOINT optimum (argmax of mean PSNR over {len(frames)} frames): (ddy,ddx)=({g_ddy:+d},{g_ddx:+d}) mean PSNR {mean_g[gi,gj]:.2f} dB vs {mean_g[ddys.index(0), ddxs.index(0)]:.2f} dB at (0,0)", flush=True)
    print(f"[{clip}] per-frame ddx range [{min(ddx_list)},{max(ddx_list)}] (spread {spread} px), ddy range [{min(ddy_list)},{max(ddy_list)}]; median ({med[0]:+d},{med[1]:+d}); "
          f"{'STABLE (<=4 px)' if spread <= 4 else 'NOT STABLE (>4 px) -- global shift is an approximation'}", flush=True)

    # --- spatial variation: 3x4 blocks of 192x256 at three frames, ddx search at the global ddy ---
    blocks = []
    for fi in BLOCK_FRAMES:
        if fi >= n: continue
        f = vr[fi].asnumpy()
        TR = torch.from_numpy(f[:H, W:2 * W].astype(np.float32))
        BR = torch.from_numpy(f[H:, W:2 * W].astype(np.float32))[top:top + TH, left:left + TW]
        hole = torch.from_numpy(f[H:, :W].astype(np.float32).mean(-1)[top:top + TH, left:left + TW] > 127.5)
        for by in range(3):
            for bx in range(4):
                ys, xs = slice(by * 192, (by + 1) * 192), slice(bx * 256, (bx + 1) * 256)
                v = (~hole[ys, xs]).float().unsqueeze(-1); nv = float(v.sum().item()) * 3
                if nv < 0.5 * 192 * 256 * 3: continue
                best = (-1e9, None); p0 = None
                for ddx in ddxs:
                    t, l = top + g_ddy, left + ddx
                    d = TR[t:t + TH, l:l + TW][ys, xs] - BR[ys, xs]
                    p = psnr_from_mse(float((d * d * v).sum().item()) / nv)
                    if ddx == 0: p0 = p
                    if p > best[0]: best = (p, ddx)
                blocks.append(dict(frame=fi, by=by, bx=bx, ddx=best[1], psnr_best=best[0], psnr_zero=p0, gain=best[0] - p0))
    informative = [b for b in blocks if b["gain"] >= 1.0]
    bx_list = [b["ddx"] for b in informative]
    if bx_list:
        print(f"[{clip}] per-block (192x256) ddx at ddy={g_ddy:+d}, blocks with >=1 dB gain: {len(informative)}/{len(blocks)}; ddx range [{min(bx_list)},{max(bx_list)}], median {int(np.median(bx_list)):+d}, "
              f"IQR [{int(np.percentile(bx_list,25)):+d},{int(np.percentile(bx_list,75)):+d}]", flush=True)
    for b in blocks:
        print(f"    frame {b['frame']:3d} block ({b['by']},{b['bx']}): ddx {b['ddx']:+d} psnr {b['psnr_best']:.2f} (at 0: {b['psnr_zero']:.2f}, gain {b['gain']:+.2f}){'' if b['gain']>=1.0 else '  [uninformative]'}", flush=True)

    # --- apply the clip-global shift to ALL frames, fill exposed pixels from BR, write FFV1 ---
    reg = np.empty((n, TH, TW, 3), np.uint8); n_filled = 0
    ys = top + g_ddy + np.arange(TH); xs = left + g_ddx + np.arange(TW)
    yv = (ys >= 0) & (ys < H); xv = (xs >= 0) & (xs < W)
    psnr_reg = []; psnr_unreg = []
    CH = 16
    for s in range(0, n, CH):
        fb = vr.get_batch(list(range(s, min(s + CH, n)))).asnumpy()
        for k in range(fb.shape[0]):
            f = fb[k]; TR = f[:H, W:2 * W]; BRw = f[H:, W:2 * W][top:top + TH, left:left + TW]
            r = BRw.copy()
            r[np.ix_(np.where(yv)[0], np.where(xv)[0])] = TR[np.ix_(ys[yv], xs[xv])]
            n_filled += int((~yv).sum() * TW + yv.sum() * (~xv).sum())
            reg[s + k] = r
            hole = f[H:, :W].astype(np.float32).mean(-1)[top:top + TH, left:left + TW] > 127.5
            v = (~hole)[..., None].astype(np.float64); nv = v.sum() * 3
            d1 = (r.astype(np.float64) - BRw) ** 2; d0 = (TR[top:top + TH, left:left + TW].astype(np.float64) - BRw) ** 2
            psnr_reg.append(psnr_from_mse((d1 * v).sum() / nv)); psnr_unreg.append(psnr_from_mse((d0 * v).sum() / nv))
    digest = hashlib.md5(reg.tobytes()).hexdigest()
    out_mkv = os.path.join(HERE, f"{clip}_gt_registered_576x1024.mkv")
    assert not os.path.exists(out_mkv), out_mkv
    IL._ffv1_write(reg, fps, out_mkv)
    with open(out_mkv + ".md5", "w") as fh: fh.write(f"{digest}  {tuple(reg.shape)}  fps={fps:.6f}  shift(ddy,ddx)=({g_ddy},{g_ddx})\n")
    dec = VideoReader(out_mkv, ctx=cpu(0)); dec_arr = dec.get_batch(list(range(len(dec)))).asnumpy()
    dec_md5 = hashlib.md5(np.ascontiguousarray(dec_arr).tobytes()).hexdigest()
    print(f"[{clip}] registered GT written {out_mkv} shape {reg.shape} pre-encode md5 {digest} decoded md5 {dec_md5} BIT_EXACT={dec_md5 == digest} filled-from-BR pixels {n_filled} (of {n*TH*TW})", flush=True)
    print(f"[{clip}] whole-clip mean PSNR(GT vs BR, holes excluded): unregistered {np.mean(psnr_unreg):.2f} dB -> registered {np.mean(psnr_reg):.2f} dB", flush=True)
    assert dec_md5 == digest

    # --- inspection PNGs: BR | unregistered TR | registered TR, and |diff| maps ---
    for fi in BLOCK_FRAMES:
        if fi >= n: continue
        f = vr[fi].asnumpy(); TRw = f[:H, W:2 * W][top:top + TH, left:left + TW]; BRw = f[H:, W:2 * W][top:top + TH, left:left + TW]
        top_row = np.concatenate([BRw, TRw, reg[fi]], axis=1)
        d_un = np.abs(TRw.astype(np.int16) - BRw.astype(np.int16)).clip(0, 255).astype(np.uint8)
        d_re = np.abs(reg[fi].astype(np.int16) - BRw.astype(np.int16)).clip(0, 255).astype(np.uint8)
        bot_row = np.concatenate([np.zeros_like(BRw), d_un, d_re], axis=1)
        panel = np.concatenate([top_row, bot_row], axis=0)
        cv2.putText(panel, f"{clip} f{fi}: BR (warped input) | real TR unregistered | real TR registered ({g_ddy:+d},{g_ddx:+d}); bottom: |TR-BR| unreg / reg", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
        cv2.imwrite(os.path.join(HERE, f"{clip}_inspect_f{fi:03d}.png"), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))

    res = dict(clip=clip, n_frames=n, fps=fps, quadrant=[H, W], window_top_left=[top, left], search=dict(ddy=DDY, ddx=DDX),
               frames_sampled=frames, per_frame=per_frame, joint_optimum=dict(ddy=g_ddy, ddx=g_ddx, mean_psnr=float(mean_g[gi, gj]), mean_psnr_zero=float(mean_g[ddys.index(0), ddxs.index(0)])),
               per_frame_ddx_range=[min(ddx_list), max(ddx_list)], per_frame_ddx_spread=spread, stable_le4px=bool(spread <= 4), median=med,
               blocks=blocks, block_ddx_range_informative=([min(bx_list), max(bx_list)] if bx_list else None),
               applied_shift=dict(ddy=g_ddy, ddx=g_ddx), filled_from_BR_pixels=n_filled,
               registered_mkv=out_mkv, pre_encode_md5=digest, decoded_md5=dec_md5,
               mean_psnr_unreg_allframes=float(np.mean(psnr_unreg)), mean_psnr_reg_allframes=float(np.mean(psnr_reg)),
               mask_disagree_frac_frame0=float((m_train != m_splat).mean()), seconds=time.time() - t_all)
    np.save(os.path.join(HERE, f"{clip}_psnr_grid_frames_x_ddy_x_ddx.npy"), G)
    return res

if __name__ == "__main__":
    clips = sys.argv[1:] or ["0301", "0204"]
    allres = {}
    for c in clips:
        allres[c] = run_clip(c)
    outj = os.path.join(HERE, "registration.json")
    json.dump(allres, open(outj, "w"), indent=1)
    print("REGISTRATION_DONE", outj, flush=True)
