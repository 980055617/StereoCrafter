"""EXPLORATORY ONLY (not the pre-registered fit): rough CPU scan of PSNR(real right eye, re-splat at scale s) vs s,
5 frames, GT shift ddx step 2 in [-240,80], ddy in {-1,0}, no common support (holes of each s excluded).
usage: fit_rough_cpu_v1.py <clip>"""
import sys, os, json, math
import numpy as np, torch
from decord import VideoReader, cpu
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import splatlib as S
torch.set_num_threads(4)
clip = sys.argv[1]
P = f"/mnt/ssd_data/deep_20261004/input_side/prep_v2/{clip}"
m = json.load(open(f"{P}/meta.json")); top, lft, Wq = m["top"], m["lft"], m["Wq"]
DEP = np.load(f"{P}/depth_rows.npy", mmap_mode="r"); LR = np.load(f"{P}/left_rows.npy", mmap_mode="r")
vt = VideoReader(f"{S.REPO}/video_data/train/{clip}_train.mp4", ctx=cpu(0))
fr = [20, 50, 80, 110, 140]
TR = {}
for i in range(max(fr) + 1):
    f = vt.next().asnumpy()
    if i in fr:
        TR[i] = f[top - 1:top + 577, Wq + lft - 240:Wq + lft + 1024 + 80].astype(np.float32)
for s in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0):
    tot = {}
    for i in fr:
        left = torch.from_numpy(np.asarray(LR[i]).astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()
        disp = torch.from_numpy((np.asarray(DEP[i]) * 2 - 1) * 20.0 * s).float()
        out, cov, _ = S.splat_rows(left, disp)
        B = np.floor(np.clip(out.permute(1, 2, 0).numpy() * 255, 0, 255))[:, lft:lft + 1024]
        v = ((1 - cov.clamp(0, 1)).numpy()[:, lft:lft + 1024] <= 0.5)[..., None]
        for ddy in (-1, 0):
            for ddx in range(-240, 81, 2):
                G = TR[i][1 + ddy:1 + ddy + 576, 240 + ddx:240 + ddx + 1024]
                e = (((G - B) ** 2) * v).sum() / (v.sum() * 3)
                tot.setdefault((ddy, ddx), []).append(e)
    best = min(tot.items(), key=lambda kv: np.mean(kv[1]))
    print(f"{clip} s={s:.2f} best shift {best[0]} PSNR {10*math.log10(255**2/np.mean(best[1])):.3f}", flush=True)
