"""Feasibility v2: depth recovery variants (raw k / LS inversion of the bilinear upsampling / area down-up / median)
-> re-splat (bilinear, base 1.414) -> agreement with the deployed BR quadrant at the deployed window.
usage: feas_v2.py <clip> <leftsrc> <frame> [<frame> ...]      (CPU only)"""
import sys, time
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from decord import VideoReader, cpu
sys.path.insert(0, "/home/kawa/master_project/StereoCrafter/scripts/distill/runs/deep_20261004/input_side")
import splatlib as S
torch.set_num_threads(8)
clip, leftsrc = sys.argv[1], sys.argv[2]
frames = [int(x) for x in sys.argv[3:]]
vs = VideoReader(f"{S.REPO}/video_data/splatting/{clip}_splatting_results.mp4", ctx=cpu(0))
vl = VideoReader(f"{S.REPO}/{leftsrc}", ctx=cpu(0))
for fi in frames:
    f = vs[fi].asnumpy(); Hq, Wq = f.shape[0] // 2, f.shape[1] // 2
    TL, TR, BL, BR = f[:Hq, :Wq], f[:Hq, Wq:], f[Hq:, :Wq], f[Hq:, Wq:]
    top, lft = S.window_rows_cols(Hq, Wq)
    L = vl[fi].asnumpy()
    M = 24
    rows = slice(top - M, top + S.TH + M)
    k, dist = S.invert_inferno(TR)
    kf = k.astype(np.float32)
    h, w = S.depthcrafter_proc_size(Hq, Wq)
    variants = {}
    variants["raw"] = (kf + 0.5) / 255.0
    t0 = time.time()
    variants["ls60"], _ = S.depth_ls_from_k(k, Hq, Wq, iters=60, rows=rows)
    tls = time.time() - t0
    o = torch.from_numpy((kf + 0.5) / 255.0)
    variants["areaup"] = F.interpolate(F.interpolate(o[None, None], size=(h, w), mode="area"), size=(Hq, Wq),
                                       mode="bilinear", align_corners=False)[0, 0].numpy()
    variants["med3"] = (cv2.medianBlur(k.astype(np.uint8), 3).astype(np.float32) + 0.5) / 255.0
    left = torch.from_numpy(L[top:top + S.TH].astype(np.float64) / 255.0).float().permute(2, 0, 1).contiguous()
    cs = slice(lft, lft + S.TW)
    brw, blw = BR[top:top + S.TH][:, cs], BL[top:top + S.TH][:, cs]
    hole_dep = blw.astype(np.float32).mean(-1) > 127.5
    line = f"{clip} f{fi} (LS {tls:.1f}s): "
    for name, d in variants.items():
        dd = d[top:top + S.TH]
        disp = torch.from_numpy((dd * 2.0 - 1.0) * S.MAX_DISP).float()
        out, cov, _ = S.splat_rows(left, disp)
        W8 = S.to_u8(out.permute(1, 2, 0).numpy())[:, cs]
        M8 = S.to_u8(np.repeat((1.0 - cov.clamp(0, 1)).numpy()[..., None], 3, -1))[:, cs]
        hole_me = M8.astype(np.float32).mean(-1) > 127.5
        valid = ~(hole_dep | hole_me)
        maskmad = np.abs(M8.astype(np.float32) - blw.astype(np.float32)).mean()
        line += (f"| {name}: PSNR {S.psnr_u8(W8, brw, valid):.2f} holes {hole_me.mean()*100:.2f}% (dep {hole_dep.mean()*100:.2f}%)"
                 f" dis {(hole_dep != hole_me).mean()*100:.3f}% maskMAD {maskmad:.2f} ")
    print(line, flush=True)
