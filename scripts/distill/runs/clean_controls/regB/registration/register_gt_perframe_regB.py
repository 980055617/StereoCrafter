"""Optional refinement for the clip whose global shift was NOT stable (0301, per-frame ddx spread 9 px):
exhaustive per-frame (ddy, ddx) for EVERY frame (same search as register_gt_regB.py), temporally median-filtered
(window 5) to remove single-frame jitter, then applied frame by frame.  Output:
  <clip>_gt_registered_perframe_576x1024.mkv (+ .md5), registration_perframe.json
Nothing tracked is modified.  usage: python register_gt_perframe_regB.py 0301
"""
import os, sys, json, hashlib, time
import numpy as np, torch
from decord import VideoReader, cpu
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import register_gt_regB as R   # crop_window, masked_psnr_grid, psnr_from_mse, IL (ffv1 writer)
TH, TW = R.TH, R.TW

def run(clip):
    t_all = time.time()
    vr = VideoReader(f"video_data/train/{clip}_train.mp4", ctx=cpu(0)); n = len(vr); fps = float(vr.get_avg_fps())
    f0 = vr[0].asnumpy(); H, W = f0.shape[0] // 2, f0.shape[1] // 2; top, left = R.crop_window(H, W)
    ddys = list(range(-R.DDY, R.DDY + 1)); ddxs = list(range(-R.DDX, R.DDX + 1))
    raw = []
    for fi in range(n):
        f = vr[fi].asnumpy()
        TR = torch.from_numpy(f[:H, W:2 * W].astype(np.float32)); BR = torch.from_numpy(f[H:, W:2 * W].astype(np.float32))[top:top + TH, left:left + TW]
        valid = ~torch.from_numpy(f[H:, :W].astype(np.float32).mean(-1)[top:top + TH, left:left + TW] > 127.5)
        g = R.masked_psnr_grid(TR, BR, valid, top, left, ddys, ddxs); bi, bj = np.unravel_index(np.nanargmax(g), g.shape)
        raw.append((ddys[bi], ddxs[bj], float(g[bi, bj]), float(g[ddys.index(0), ddxs.index(0)])))
        if fi % 10 == 0: print(f"[{clip}] frame {fi:3d}: ({ddys[bi]:+d},{ddxs[bj]:+d}) {g[bi,bj]:.2f} dB (zero {g[ddys.index(0), ddxs.index(0)]:.2f})", flush=True)
    ddy_raw = np.array([r[0] for r in raw]); ddx_raw = np.array([r[1] for r in raw])
    def medfilt(a, k=5):
        p = k // 2; ap = np.pad(a, (p, p), mode="edge"); return np.array([int(np.median(ap[i:i + k])) for i in range(len(a))])
    ddy_s, ddx_s = medfilt(ddy_raw), medfilt(ddx_raw)
    print(f"[{clip}] raw ddx range [{ddx_raw.min()},{ddx_raw.max()}], smoothed range [{ddx_s.min()},{ddx_s.max()}]; frames where smoothed != raw: {(ddx_s != ddx_raw).sum()}", flush=True)
    reg = np.empty((n, TH, TW, 3), np.uint8); psnr_pf = []; n_filled = 0
    for s in range(0, n, 16):
        fb = vr.get_batch(list(range(s, min(s + 16, n)))).asnumpy()
        for k in range(fb.shape[0]):
            fi = s + k; f = fb[k]; TR = f[:H, W:2 * W]; BRw = f[H:, W:2 * W][top:top + TH, left:left + TW]
            ys = top + int(ddy_s[fi]) + np.arange(TH); xs = left + int(ddx_s[fi]) + np.arange(TW)
            yv = (ys >= 0) & (ys < H); xv = (xs >= 0) & (xs < W); r = BRw.copy()
            r[np.ix_(np.where(yv)[0], np.where(xv)[0])] = TR[np.ix_(ys[yv], xs[xv])]; n_filled += int((~yv).sum() * TW + yv.sum() * (~xv).sum()); reg[fi] = r
            hole = f[H:, :W].astype(np.float32).mean(-1)[top:top + TH, left:left + TW] > 127.5; v = (~hole)[..., None].astype(np.float64)
            psnr_pf.append(R.psnr_from_mse((((r.astype(np.float64) - BRw) ** 2) * v).sum() / (v.sum() * 3)))
    digest = hashlib.md5(reg.tobytes()).hexdigest(); out = os.path.join(HERE, f"{clip}_gt_registered_perframe_576x1024.mkv"); assert not os.path.exists(out)
    R.IL._ffv1_write(reg, fps, out); open(out + ".md5", "w").write(f"{digest}  {tuple(reg.shape)}  fps={fps:.6f}  per-frame shifts (median-5 smoothed)\n")
    dec = VideoReader(out, ctx=cpu(0)); dm = hashlib.md5(np.ascontiguousarray(dec.get_batch(list(range(len(dec)))).asnumpy()).tobytes()).hexdigest()
    print(f"[{clip}] per-frame registered GT {out} md5 {digest} decoded {dm} BIT_EXACT={dm == digest} filled {n_filled}; whole-clip mean PSNR vs BR: {np.mean(psnr_pf):.2f} dB (global-shift version: see registration.json)", flush=True)
    assert dm == digest
    res = dict(clip=clip, n_frames=n, per_frame_raw=[dict(frame=i, ddy=int(r[0]), ddx=int(r[1]), psnr_best=r[2], psnr_zero=r[3]) for i, r in enumerate(raw)],
               ddy_smoothed=[int(x) for x in ddy_s], ddx_smoothed=[int(x) for x in ddx_s], mkv=out, pre_encode_md5=digest, decoded_md5=dm,
               mean_psnr_reg_perframe=float(np.mean(psnr_pf)), filled_from_BR_pixels=n_filled, seconds=time.time() - t_all)
    outj = os.path.join(HERE, "registration_perframe.json"); old = json.load(open(outj)) if os.path.exists(outj) else {}; old[clip] = res; json.dump(old, open(outj, "w"), indent=1)
    print("PERFRAME_DONE", outj, flush=True)

if __name__ == "__main__":
    for c in (sys.argv[1:] or ["0301"]): run(c)
