"""Feasibility: can the deployed warped input be reproduced from the inferno depth tile + the source left video?
usage: feas_v1.py <clip> <leftsrc> <frame> [<frame> ...]      (CPU only)"""
import sys, time
import numpy as np
import torch
from decord import VideoReader, cpu
sys.path.insert(0, "/home/kawa/master_project/StereoCrafter/scripts/distill/runs/deep_20261004/input_side")
import splatlib as S
torch.set_num_threads(8)
clip, leftsrc = sys.argv[1], sys.argv[2]
frames = [int(x) for x in sys.argv[3:]]
vs = VideoReader(f"{S.REPO}/video_data/splatting/{clip}_splatting_results.mp4", ctx=cpu(0))
vl = VideoReader(f"{S.REPO}/{leftsrc}", ctx=cpu(0))
print(f"splat n={len(vs)} left n={len(vl)}")
for fi in frames:
    t0 = time.time()
    f = vs[fi].asnumpy(); Hq, Wq = f.shape[0] // 2, f.shape[1] // 2
    TL, TR, BL, BR = f[:Hq, :Wq], f[:Hq, Wq:], f[Hq:, :Wq], f[Hq:, Wq:]
    top, lft = S.window_rows_cols(Hq, Wq)
    L = vl[fi].asnumpy()
    assert L.shape[:2] == (Hq, Wq), (L.shape, Hq, Wq)
    rows = slice(top, top + S.TH)
    k, dist = S.invert_inferno(TR[rows])
    d = (k.astype(np.float32) + 0.5) / 255.0
    disp = torch.from_numpy((d * 2.0 - 1.0) * S.MAX_DISP).float()
    left = torch.from_numpy(L[rows].astype(np.float64) / 255.0).float().permute(2, 0, 1).contiguous()
    out, cov, wsum = S.splat_rows(left, disp)
    W8 = S.to_u8(out.permute(1, 2, 0).numpy())
    M8 = S.to_u8(np.repeat((1.0 - cov.clamp(0, 1)).numpy()[..., None], 3, -1))
    cs = slice(lft, lft + S.TW)
    brw, blw = BR[rows][:, cs], BL[rows][:, cs]
    hole_dep = blw.astype(np.float32).mean(-1) > 127.5
    hole_me = M8[:, cs].astype(np.float32).mean(-1) > 127.5
    valid = ~(hole_dep | hole_me)
    p_br = S.psnr_u8(W8[:, cs], brw, valid)
    p_tl = S.psnr_u8(L[rows][:, cs], TL[rows][:, cs])
    # the codec-noise yardstick: TL (mp4v copy of the left) vs the source left
    print(f"{clip} f{fi}: window rows {top}..{top+S.TH} cols {lft}..{lft+S.TW}  "
          f"PSNR(resplat, deployed BR | non-hole) {p_br:.2f} dB   PSNR(TL, source left) {p_tl:.2f} dB  "
          f"holes deployed {hole_dep.mean()*100:.2f}% mine {hole_me.mean()*100:.2f}% disagree {(hole_dep!=hole_me).mean()*100:.3f}%  "
          f"inferno residual median {np.median(np.sqrt(dist)):.2f} p99 {np.quantile(np.sqrt(dist),0.99):.2f}  "
          f"depth k range {k.min()}..{k.max()}  {time.time()-t0:.1f}s", flush=True)
