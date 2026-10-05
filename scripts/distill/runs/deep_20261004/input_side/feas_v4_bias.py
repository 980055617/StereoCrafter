"""Feasibility v4: residual disparity bias of the recovered depth (round-trip LUT + LS) against the deployed BR.
usage: feas_v4_bias.py <clip> <leftsrc> <frame> [...]   (CPU)"""
import sys
import numpy as np
import torch
from decord import VideoReader, cpu
sys.path.insert(0, "/home/kawa/master_project/StereoCrafter/scripts/distill/runs/deep_20261004/input_side")
import splatlib as S
torch.set_num_threads(8)
clip, leftsrc = sys.argv[1], sys.argv[2]
vs = VideoReader(f"{S.REPO}/video_data/splatting/{clip}_splatting_results.mp4", ctx=cpu(0))
vl = VideoReader(f"{S.REPO}/{leftsrc}", ctx=cpu(0))
for fi in [int(x) for x in sys.argv[3:]]:
    f = vs[fi].asnumpy(); Hq, Wq = f.shape[0] // 2, f.shape[1] // 2
    TL, TR, BL, BR = f[:Hq, :Wq], f[:Hq, Wq:], f[Hq:, :Wq], f[Hq:, Wq:]
    top, lft = S.window_rows_cols(Hq, Wq)
    L = vl[fi].asnumpy()
    cs = slice(lft, lft + S.TW)
    brw, blw = BR[top:top + S.TH][:, cs], BL[top:top + S.TH][:, cs]
    hole_dep = blw.astype(np.float32).mean(-1) > 127.5
    left = torch.from_numpy(L[top:top + S.TH].astype(np.float64) / 255.0).float().permute(2, 0, 1).contiguous()
    line = f"{clip} f{fi}: TLvsSrc {S.psnr_u8(TL[top:top+S.TH][:, cs], L[top:top+S.TH][:, cs]):.2f} |"
    for lutname in ("written", "roundtrip"):
        k, _ = S.invert_inferno(TR, lut=lutname)
        d, _ = S.depth_ls_from_k(k, Hq, Wq, iters=60, rows=slice(top - 24, top + S.TH + 24))
        base = (d[top:top + S.TH] * 2.0 - 1.0) * S.MAX_DISP
        res = []
        for off in (-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3):
            out, cov, _ = S.splat_rows(left, torch.from_numpy(base + off).float())
            W8 = S.to_u8(out.permute(1, 2, 0).numpy())[:, cs]
            hole_me = (1.0 - cov.clamp(0, 1)).numpy()[:, cs] > 0.5
            valid = ~(hole_dep | hole_me)
            res.append((S.psnr_u8(W8, brw, valid), off, (hole_dep != hole_me).mean() * 100))
        b = max(res)
        z = [r for r in res if r[1] == 0.0][0]
        line += f" {lutname}: @0 {z[0]:.2f}dB dis {z[2]:.2f}% | best {b[0]:.2f}dB @{b[1]:+.1f}px dis {b[2]:.2f}% |"
    print(line, flush=True)
