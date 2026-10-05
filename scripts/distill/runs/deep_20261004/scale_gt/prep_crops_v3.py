#!/usr/bin/env python
"""v3 = v2 + per-window diagnostic arrays lshift.npy int8 [14,2,72,128] (best local dy*,dx* in px), cellinfo.npy uint8 [14,72,128]
(bit0 info, bit1 flat, bit2 consistent) -- the valid mask itself is computed EXACTLY as in v2.
v2 = v1 with the global search widened to ddx in [-240, 40] (train clip 0094 hit v1's -120 boundary in the smoke).
scale_gt lane, PHASE A (CPU only, no GPU): registered training crops for the GT-supervised scale run.

Per clip (train tile video_data/train/<clip>_train.mp4 = 2x2 tile  TL real left | TR REAL right ; BL hole mask | BR warped):
 1. windows: 14-frame windows on the deployed stride-11 grid (starts 0, 11, ..., <= n-14, as minift/xcheck_mini_ft.py);
    NWIN=4 of them, evenly spaced (round(linspace(0, K-1, 4))).
 2. REGISTRATION (one robust global integer shift per clip, model-independent): maximise the MEAN over sampled frames of the
    masked PSNR between TR[top+ddy : +576, left+ddx : +1024] and BR[top : +576, left : +1024], holes (BL mean > 127.5)
    excluded, over ddy in [-10,10], ddx in [-240,40] (eval_robustness grid widened left: some train clips exceed -120).  Coarse-to-fine: 4x area-downsampled grid
    at 4-px steps over ddy in [-12,12] x ddx in [-240,40], then full-resolution exhaustive refinement +-4 px around the
    coarse optimum (clipped to the final range).  Sampled frames: offsets 0 and 7 of every window (8 frames).
    Boundary hit (|ddy|=10 or ddx in {-240,40}) -> clip flagged (and excluded by the trainer, PREREG).
 3. VALIDITY MASK on the 72x128 latent grid, per frame (local-shift consistency, NOT a |TR-BR| threshold):
    half-resolution, for every latent cell a 32x32-px block (centred on the cell) over NON-HOLE pixels; local search of the
    registered TR around the global shift, dy in [-4,4] px, dx in [-16,16] px (2-px steps).
      info       = >= 50 % of the block is non-hole
      flat       = median_over_shifts(E) - min(E) < (3/255)^2       (texture too weak to tell shifts apart -> valid)
      consistent = best local shift within +-2 px of the global (|dy*|,|dx*| <= 1 half-res unit) OR E(global) <= 1.1 min(E)
      info cells: valid = flat or consistent
      no-info (hole-dominated) cells inherit: invalid iff an invalid info cell lies within 2 cells (16 px)
      finally invalid cells are dilated by one cell (3x3).
 4. writes LOSSLESS uint8 arrays per window to <out_root>/crops/<clip>/w<start:03d>/:
      cond.npy [14,576,1024,3] = BR crop (model input)     bl.npy [14,576,1024,3] = BL crop (mask, all 3 channels)
      tgt.npy  [14,576,1024,3] = REGISTERED TR crop        valid.npy [14,72,128] bool
    + <out_root>/crops/<clip>/clip.json (registration, per-window/per-frame stats, md5s of the arrays).
usage: python prep_crops_v1.py <out_root> <shard> <nshards> <clip_list_json_or_comma_list> [--reg-only]
   --reg-only: registration only (no crops; for the estimator gate on regB anchors / eval_robustness REG_CLIP)
Nothing tracked is modified; nothing under video_data/ is written; an existing clip dir is never overwritten.
"""
import hashlib, json, math, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from decord import VideoReader, cpu

REPO = "/home/kawa/master_project/StereoCrafter"; os.chdir(REPO)
torch.set_num_threads(int(os.environ.get("PREP_THREADS", "4")))
TH, TW = 576, 1024
NF, STRIDE, NWIN = 14, 11, int(os.environ.get("PREP_NWIN", "4"))
DDY_LO, DDY_HI, DDX_LO, DDX_HI = -10, 10, -240, 40        # final search range (eval_robustness grid widened to -240)
MY, MXL, MXR = 16, 256, 56                                 # TR region margins: final range + validity margin (4 px y, 16 px x), multiples of 4
VD_Y, VD_X = 2, 8                                          # validity local search, half-res units (+-4 px, +-16 px)
TAU_FLAT = (3.0 / 255.0) ** 2
CONS_RATIO = 1.1
P255 = 20 * math.log10(255.0)

OUT_ROOT, SHARD, NSH, CL = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
REG_ONLY = "--reg-only" in sys.argv
clips = json.load(open(CL)) if CL.endswith(".json") else CL.split(",")
if isinstance(clips, dict): clips = clips["clips"]
clips = [c for i, c in enumerate(clips) if i % NSH == SHARD]


def crop_window(H, W):
    h, w = H // 128 * 128, W // 128 * 128
    return (h - TH) // 2, (w - TW) // 2


def windows_for(n):
    starts = list(range(0, n - NF + 1, STRIDE))
    idx = sorted(set(int(round(x)) for x in np.linspace(0, len(starts) - 1, NWIN)))
    return [starts[i] for i in idx]


def pool(x, k):   # x [H,W,C] or [H,W] float tensor -> area-downsampled by k
    if x.dim() == 2:
        return F.avg_pool2d(x[None, None], k)[0, 0]
    return F.avg_pool2d(x.permute(2, 0, 1)[None], k)[0].permute(1, 2, 0)


def psnr(sse, n):
    return P255 - 10 * math.log10(max(sse / max(n, 1.0), 1e-12))


def reg_grids(frames_parts):
    """frames_parts: list of (TRr uint8 [TH+2MY, TW+MXL+MXR, 3], BRw uint8 [TH,TW,3], hole bool [TH,TW]).
    Returns coarse mean-PSNR grid, coarse optimum, fine mean grid, fine optimum, per-frame fine optima, psnr at 0."""
    cys = list(range(-12, 13, 4)); cxs = list(range(DDX_LO, DDX_HI + 1, 4))
    G = np.zeros((len(frames_parts), len(cys), len(cxs)))
    for k, (TRr, BRw, hole) in enumerate(frames_parts):
        T4 = pool(torch.from_numpy(TRr).float(), 4); B4 = pool(torch.from_numpy(BRw).float(), 4)
        v4 = (pool(torch.from_numpy(~hole).float(), 4) >= 0.999).float().unsqueeze(-1); nv = float(v4.sum()) * 3
        for i, sy in enumerate(cys):
            oy = (MY + sy) // 4
            for j, sx in enumerate(cxs):
                ox = (MXL + sx) // 4
                d = T4[oy:oy + TH // 4, ox:ox + TW // 4] - B4
                G[k, i, j] = psnr(float((d * d * v4).sum()), nv)
    Gm = G.mean(0); ci, cj = np.unravel_index(int(np.argmax(Gm)), Gm.shape)
    cy, cx = cys[ci], cxs[cj]
    fys = list(range(max(cy - 4, DDY_LO), min(cy + 4, DDY_HI) + 1)); fxs = list(range(max(cx - 4, DDX_LO), min(cx + 4, DDX_HI) + 1))
    FG = np.zeros((len(frames_parts), len(fys), len(fxs))); P0 = []
    for k, (TRr, BRw, hole) in enumerate(frames_parts):
        T = torch.from_numpy(TRr).float(); B = torch.from_numpy(BRw).float()
        v = torch.from_numpy(~hole).float().unsqueeze(-1); nv = float(v.sum()) * 3
        for i, sy in enumerate(fys):
            for j, sx in enumerate(fxs):
                d = T[MY + sy:MY + sy + TH, MXL + sx:MXL + sx + TW] - B
                FG[k, i, j] = psnr(float((d * d * v).sum()), nv)
        d0 = T[MY:MY + TH, MXL:MXL + TW] - B; P0.append(psnr(float((d0 * d0 * v).sum()), nv))
    FGm = FG.mean(0); fi, fj = np.unravel_index(int(np.argmax(FGm)), FGm.shape)
    per_frame = []
    for k in range(len(frames_parts)):
        a, b = np.unravel_index(int(np.argmax(FG[k])), FG[k].shape)
        per_frame.append(dict(ddy=fys[a], ddx=fxs[b], psnr=float(FG[k, a, b]), at_fine_edge=bool(a in (0, len(fys) - 1) or b in (0, len(fxs) - 1))))
    return dict(coarse=dict(ddy=cy, ddx=cx, mean_psnr=float(Gm[ci, cj])),
                ddy=fys[fi], ddx=fxs[fj], mean_psnr=float(FGm[fi, fj]), mean_psnr_zero=float(np.mean(P0)),
                coarse_mean_psnr_zero=float(Gm[cys.index(0), cxs.index(0)]), per_frame=per_frame,
                boundary=bool(abs(fys[fi]) == 10 or fxs[fj] in (DDX_LO, DDX_HI)))


def validity(TRm, BRw, hole):
    """TRm uint8 [TH+2*2*VD_Y, TW+2*2*VD_X, 3] = registered TR with a (4 px, 16 px) margin; BRw uint8; hole bool.
    Returns valid [72,128] bool and diagnostics."""
    T2 = pool(torch.from_numpy(TRm).float() / 255.0, 2); B2 = pool(torch.from_numpy(BRw).float() / 255.0, 2)
    nh2 = (pool(torch.from_numpy(~hole).float(), 2) >= 0.999).float()
    h2, w2 = TH // 2, TW // 2
    shifts = [(dy, dx) for dy in range(-VD_Y, VD_Y + 1) for dx in range(-VD_X, VD_X + 1)]
    E = torch.empty(len(shifts), h2, w2)
    for s, (dy, dx) in enumerate(shifts):
        d = T2[VD_Y + dy:VD_Y + dy + h2, VD_X + dx:VD_X + dx + w2] - B2
        E[s] = (d * d).mean(-1) * nh2
    Es = F.avg_pool2d(E[:, None], 16, 4, padding=6, count_include_pad=True)[:, 0]          # block mean incl. zeros
    N = F.avg_pool2d(nh2[None, None], 16, 4, padding=6, count_include_pad=True)[0, 0]       # informative fraction
    Em = Es / N.clamp(min=1e-6)
    i0 = shifts.index((0, 0))
    E0 = Em[i0]; Emin, amin = Em.min(0); Emed = Em.median(0).values
    dys = torch.tensor([s[0] for s in shifts])[amin]; dxs = torch.tensor([s[1] for s in shifts])[amin]
    info = N >= 0.5
    flat = (Emed - Emin) < TAU_FLAT
    consistent = ((dys.abs() <= 1) & (dxs.abs() <= 1)) | (E0 <= CONS_RATIO * Emin + 1e-9)
    inv_info = info & ~(flat | consistent)
    near_inv = F.max_pool2d(inv_info.float()[None, None], 5, 1, 2)[0, 0] > 0
    valid = torch.where(info, ~inv_info, ~near_inv)
    valid = ~(F.max_pool2d((~valid).float()[None, None], 3, 1, 1)[0, 0] > 0)
    assert valid.shape == (TH // 8, TW // 8), valid.shape
    diag = dict(lshift=torch.stack([dys * 2, dxs * 2]).to(torch.int8).numpy(),
                cellinfo=(info.to(torch.uint8) | (flat.to(torch.uint8) << 1) | (consistent.to(torch.uint8) << 2)).numpy())
    return valid.numpy(), dict(info=float(info.float().mean()), flat=float((info & flat).float().mean()),
                               inv_info=float(inv_info.float().mean()), kept=float(valid.float().mean()), diag=diag)


def md5(a):
    return hashlib.md5(np.ascontiguousarray(a).tobytes()).hexdigest()


for clip in clips:
    t0 = time.time()
    cdir = os.path.join(OUT_ROOT, "crops" if not REG_ONLY else "reg_only", clip)
    if os.path.exists(os.path.join(cdir, "clip.json")):
        print(f"[{clip}] SKIP exists {cdir}", flush=True); continue
    assert not os.path.exists(cdir), f"partial dir exists, not overwritten: {cdir}"
    p = f"video_data/train/{clip}_train.mp4"
    assert int(clip) < 310 and "train_leftGT_broken" not in os.path.realpath(p), clip
    vr = VideoReader(p, ctx=cpu(0)); n = len(vr)
    f0 = vr[0].asnumpy(); H, W = f0.shape[0] // 2, f0.shape[1] // 2
    top, left = crop_window(H, W)
    assert top - MY >= 0 and left - MXL >= 0 and top + TH + MY <= H and left + TW + MXR <= W, (clip, H, W, top, left)
    wins = windows_for(n)
    need = sorted(set(s + o for s in wins for o in range(NF)))
    frames = {}
    for s0 in range(0, len(need), 16):
        part = need[s0:s0 + 16]; b = vr.get_batch(part).asnumpy()
        for k, fi in enumerate(part):
            f = b[k]
            frames[fi] = dict(BR=f[H + top:H + top + TH, W + left:W + left + TW].copy(),
                              BL=f[H + top:H + top + TH, left:left + TW].copy(),
                              TRr=f[top - MY:top + TH + MY, W + left - MXL:W + left + TW + MXR].copy(),
                              TLc=f[top:top + TH, left:left + TW].copy())
        del b
    t_dec = time.time() - t0
    holes = {fi: frames[fi]["BL"].astype(np.float32).mean(-1) > 127.5 for fi in need}
    reg_frames = [s + o for s in wins for o in (0, 7)]
    reg = reg_grids([(frames[fi]["TRr"], frames[fi]["BR"], holes[fi]) for fi in reg_frames])
    reg.update(clip=clip, n_frames=n, quadrant=[H, W], window=[top, left], windows=wins, reg_frames=reg_frames,
               tr_tl_mad_255=float(np.abs(frames[0]["TRr"][MY:MY + TH, MXL:MXL + TW].astype(np.float32) - frames[0]["TLc"].astype(np.float32)).mean()))
    gy, gx = reg["ddy"], reg["ddx"]
    print(f"[{clip}] n={n} quad {H}x{W} win ({top},{left}) windows {wins} | REG coarse ({reg['coarse']['ddy']:+d},{reg['coarse']['ddx']:+d}) "
          f"-> ({gy:+d},{gx:+d}) meanPSNR {reg['mean_psnr']:.2f} (zero {reg['mean_psnr_zero']:.2f}) boundary={reg['boundary']} "
          f"per-frame ddx {[q['ddx'] for q in reg['per_frame']]} decode {t_dec:.1f}s", flush=True)
    if REG_ONLY:
        os.makedirs(cdir); json.dump(reg, open(os.path.join(cdir, "clip.json"), "w"), indent=1); continue
    os.makedirs(cdir)
    wstats = []
    for s in wins:
        wd = os.path.join(cdir, f"w{s:03d}"); os.makedirs(wd)
        cond = np.stack([frames[s + o]["BR"] for o in range(NF)])
        bl = np.stack([frames[s + o]["BL"] for o in range(NF)])
        tgt = np.stack([frames[s + o]["TRr"][MY + gy:MY + gy + TH, MXL + gx:MXL + gx + TW] for o in range(NF)])
        vals, vstat = [], []
        for o in range(NF):
            fi = s + o; TRr = frames[fi]["TRr"]
            m_y, m_x = 2 * VD_Y, 2 * VD_X
            TRm = TRr[MY + gy - m_y:MY + gy + TH + m_y, MXL + gx - m_x:MXL + gx + TW + m_x]
            assert TRm.shape == (TH + 2 * m_y, TW + 2 * m_x, 3), TRm.shape
            v, st = validity(TRm, frames[fi]["BR"], holes[fi]); vals.append(v); vstat.append(st)
        valid = np.stack(vals)
        hole_w = np.stack([holes[s + o] for o in range(NF)])
        nh = ~hole_w
        diff = (tgt.astype(np.float64) - cond.astype(np.float64))[nh]          # photometric offset on non-hole pixels
        mse_reg = float((diff ** 2).mean()) / 255.0 ** 2
        z = np.stack([frames[s + o]["TRr"][MY:MY + TH, MXL:MXL + TW] for o in range(NF)]).astype(np.float64)
        mse_zero = float(((z - cond.astype(np.float64))[nh] ** 2).mean()) / 255.0 ** 2
        for name, arr in (("cond", cond), ("bl", bl), ("tgt", tgt), ("valid", valid)):
            np.save(os.path.join(wd, f"{name}.npy"), arr)
        np.save(os.path.join(wd, "lshift.npy"), np.stack([x["diag"]["lshift"] for x in vstat]))
        np.save(os.path.join(wd, "cellinfo.npy"), np.stack([x["diag"]["cellinfo"] for x in vstat]))
        ws = dict(start=s, kept=float(valid.mean()), kept_per_frame=[float(x["kept"]) for x in vstat],
                  info=float(np.mean([x["info"] for x in vstat])), flat=float(np.mean([x["flat"] for x in vstat])),
                  inv_info=float(np.mean([x["inv_info"] for x in vstat])), hole_frac=float(hole_w.mean()),
                  mask_frac_minift=float((bl.astype(np.float32) / 255.0).mean()),
                  rgb_offset_255=[float(x) for x in diff.mean(0)], psnr_reg=-10 * math.log10(max(mse_reg, 1e-12)),
                  psnr_zero=-10 * math.log10(max(mse_zero, 1e-12)),
                  md5=dict(cond=md5(cond), bl=md5(bl), tgt=md5(tgt), valid=md5(valid)))
        wstats.append(ws)
        print(f"[{clip}] w{s:03d} kept {ws['kept']:.3f} (info {ws['info']:.3f} flat {ws['flat']:.3f} inv {ws['inv_info']:.3f}) hole {ws['hole_frac']:.4f} "
              f"PSNR(TR,BR) reg {ws['psnr_reg']:.2f} zero {ws['psnr_zero']:.2f} rgb_off {[round(x,1) for x in ws['rgb_offset_255']]}", flush=True)
    reg["windows_stats"] = wstats; reg["seconds"] = time.time() - t0
    reg["params"] = dict(NF=NF, STRIDE=STRIDE, NWIN=NWIN, range=[DDY_LO, DDY_HI, DDX_LO, DDX_HI], VD=[VD_Y, VD_X],
                         TAU_FLAT=TAU_FLAT, CONS_RATIO=CONS_RATIO)
    json.dump(reg, open(os.path.join(cdir, "clip.json"), "w"), indent=1)
    print(f"[{clip}] DONE {time.time()-t0:.1f}s", flush=True)
print("SHARD_DONE", SHARD, flush=True)
