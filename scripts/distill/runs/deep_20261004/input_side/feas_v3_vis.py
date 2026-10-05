"""Feasibility v3: visual diff of re-splat (LS depth) vs deployed BR + per-column-shift test (is there a systematic
sub-pixel disparity offset?).  usage: feas_v3_vis.py <clip> <leftsrc> <frame> <outdir>   (CPU)"""
import sys
import numpy as np
import torch
import cv2
from decord import VideoReader, cpu
sys.path.insert(0, "/home/kawa/master_project/StereoCrafter/scripts/distill/runs/deep_20261004/input_side")
import splatlib as S
torch.set_num_threads(8)
clip, leftsrc, fi, od = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
vs = VideoReader(f"{S.REPO}/video_data/splatting/{clip}_splatting_results.mp4", ctx=cpu(0))
vl = VideoReader(f"{S.REPO}/{leftsrc}", ctx=cpu(0))
f = vs[fi].asnumpy(); Hq, Wq = f.shape[0] // 2, f.shape[1] // 2
TL, TR, BL, BR = f[:Hq, :Wq], f[:Hq, Wq:], f[Hq:, :Wq], f[Hq:, Wq:]
top, lft = S.window_rows_cols(Hq, Wq)
L = vl[fi].asnumpy()
k, _ = S.invert_inferno(TR)
d, _ = S.depth_ls_from_k(k, Hq, Wq, iters=60, rows=slice(top - 24, top + S.TH + 24))
left = torch.from_numpy(L[top:top + S.TH].astype(np.float64) / 255.0).float().permute(2, 0, 1).contiguous()
cs = slice(lft, lft + S.TW)
brw, blw = BR[top:top + S.TH][:, cs], BL[top:top + S.TH][:, cs]
hole_dep = blw.astype(np.float32).mean(-1) > 127.5
base = (d[top:top + S.TH] * 2.0 - 1.0) * S.MAX_DISP
res = []
for off in (-0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6):
    for sc in (0.98, 1.0, 1.02):
        disp = torch.from_numpy(base * sc + off).float()
        out, cov, _ = S.splat_rows(left, disp)
        W8 = S.to_u8(out.permute(1, 2, 0).numpy())[:, cs]
        hole_me = (1.0 - cov.clamp(0, 1)).numpy()[:, cs] > 0.5
        valid = ~(hole_dep | hole_me)
        res.append((S.psnr_u8(W8, brw, valid), off, sc, (hole_dep != hole_me).mean() * 100))
res.sort(reverse=True)
for r in res[:6]:
    print(f"PSNR {r[0]:.2f} dB at disp*{r[2]} + {r[1]:+.1f} px  holeDisagree {r[3]:.3f}%")
disp = torch.from_numpy(base).float()
out, cov, _ = S.splat_rows(left, disp)
W8 = S.to_u8(out.permute(1, 2, 0).numpy())[:, cs]
M8 = S.to_u8((1.0 - cov.clamp(0, 1)).numpy())[:, cs]
diff = np.abs(W8.astype(np.int16) - brw.astype(np.int16)).max(-1).clip(0, 255).astype(np.uint8)
# local PSNR map in 32x32 blocks
e = ((W8.astype(np.float64) - brw.astype(np.float64)) ** 2).mean(-1)
B = 32
blk = e[:S.TH // B * B, :S.TW // B * B].reshape(S.TH // B, B, S.TW // B, B).mean((1, 3))
print("block PSNR quantiles (5,25,50,75,95):", np.round(10 * np.log10(255 ** 2 / np.maximum(np.quantile(blk, [0.95, 0.75, 0.5, 0.25, 0.05]), 1e-9)), 2))
y0, x0 = 200, 300
crop = lambda a: a[y0:y0 + 256, x0:x0 + 384]
gray3 = lambda a: np.repeat(a[..., None], 3, -1) if a.ndim == 2 else a
panel = np.concatenate([np.concatenate([crop(brw), crop(W8), gray3(crop(diff * 4))], 1),
                        np.concatenate([gray3(crop(blw.mean(-1).astype(np.uint8))), gray3(crop(M8)),
                                        gray3(crop((np.abs(blw.mean(-1) - M8) > 127).astype(np.uint8) * 255))], 1)], 0)
panel = cv2.resize(panel, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
cv2.imwrite(f"{od}/{clip}_f{fi:03d}_resplat_vs_deployed.png", cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
print("wrote", f"{od}/{clip}_f{fi:03d}_resplat_vs_deployed.png")
