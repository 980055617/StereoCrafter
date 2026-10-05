"""input_side lane, step P (v2 = v1 with the closed-form separable LS; v1 was stopped unfinished, its partial
outputs in prep_v1/ are kept untouched and unused): per-clip preparation (CPU only; nothing tracked is read-modified, nothing overwritten).

usage: prep_v1.py <clip> <left_source_video> <out_root>
writes <out_root>/<clip>/ (refuses if it exists):
  deployed_left.npy   uint8 [T,576,1024,3]  splatting TL at the deployed window (what the deployed reader returns)
  deployed_warped.npy uint8 [T,576,1024,3]  splatting BR at the deployed window
  deployed_mask.npy   uint8 [T,576,1024,3]  splatting BL at the deployed window
  left_rows.npy       uint8 [T,576,Wq,3]    source left video (sequential decord decode), window rows, full width
  depth_rows.npy      float32 [T,576,Wq]    recovered splatted depth d (round-trip-LUT inferno inversion with tie
                                            averaging + closed-form LS inversion of DepthCrafter's bilinear upsampling)
  meta.json, prep_log.txt  (per frame: PSNR(TL, source left) = alignment/codec yardstick; re-splat at the deployed
                            mapping vs the deployed BR (non-hole PSNR), hole disagreement)
All decoding is SEQUENTIAL (VideoReader.next()); decord random access mis-seeks on the HEVC left_eye sources.
"""
import json
import os
import sys
import time

import numpy as np
import torch
from decord import VideoReader, cpu

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import splatlib as S  # noqa: E402

torch.set_num_threads(int(os.environ.get("PREP_THREADS", "4")))
clip, leftsrc, root = sys.argv[1], sys.argv[2], sys.argv[3]
od = os.path.join(root, clip)
assert not os.path.exists(od), f"refusing to overwrite {od}"
os.makedirs(od)
MARG = 48
splat_path = f"{S.REPO}/video_data/splatting/{clip}_splatting_results.mp4"
vs = VideoReader(splat_path, ctx=cpu(0))
vl = VideoReader(f"{S.REPO}/{leftsrc}", ctx=cpu(0))
T = len(vs)
assert len(vl) >= T, (len(vl), T)
fps = float(vs.get_avg_fps())
f0 = vs.next().asnumpy()
vs = VideoReader(splat_path, ctx=cpu(0))                         # re-open: sequential from frame 0
Hq, Wq = f0.shape[0] // 2, f0.shape[1] // 2
top, lft = S.window_rows_cols(Hq, Wq)
R0, R1 = max(top - MARG, 0), min(top + S.TH + MARG, Hq)
LS = S.SeparableLS(Hq, Wq, R0, R1, top)
mm = lambda name, shape, dt: np.lib.format.open_memmap(os.path.join(od, name), mode="w+", dtype=dt, shape=shape)
A_left = mm("deployed_left.npy", (T, S.TH, S.TW, 3), np.uint8)
A_warp = mm("deployed_warped.npy", (T, S.TH, S.TW, 3), np.uint8)
A_mask = mm("deployed_mask.npy", (T, S.TH, S.TW, 3), np.uint8)
A_lrow = mm("left_rows.npy", (T, S.TH, Wq, 3), np.uint8)
A_dep = mm("depth_rows.npy", (T, S.TH, Wq), np.float32)
log = open(os.path.join(od, "prep_log.txt"), "w")
cs = slice(lft, lft + S.TW)
stats = []
t_start = time.time()
for i in range(T):
    f = vs.next().asnumpy()
    L = vl.next().asnumpy()
    assert L.shape[:2] == (Hq, Wq), (L.shape, Hq, Wq)
    A_left[i] = f[top:top + S.TH, lft:lft + S.TW]
    A_mask[i] = f[Hq + top:Hq + top + S.TH, lft:lft + S.TW]
    A_warp[i] = f[Hq + top:Hq + top + S.TH, Wq + lft:Wq + lft + S.TW]
    A_lrow[i] = L[top:top + S.TH]
    k, kd = S.invert_inferno_fast(f[R0:R1, Wq:2 * Wq])
    A_dep[i], _ = LS(k)
    # checks: alignment (TL vs source), reproduction of the deployed BR at the deployed mapping
    p_tl = S.psnr_u8(A_left[i], L[top:top + S.TH, lft:lft + S.TW])
    left = torch.from_numpy(L[top:top + S.TH].astype(np.float64) / 255.0).float().permute(2, 0, 1).contiguous()
    disp = torch.from_numpy((A_dep[i] * 2.0 - 1.0) * S.MAX_DISP).float()
    out, cov, _ = S.splat_rows(left, disp)
    w8 = S.to_u8(out.permute(1, 2, 0).numpy())[:, cs]
    hole_me = (1.0 - cov.clamp(0, 1)).numpy()[:, cs] > 0.5
    hole_dep = A_mask[i].astype(np.float32).mean(-1) > 127.5
    valid = ~(hole_dep | hole_me)
    p_br = S.psnr_u8(w8, A_warp[i], valid)
    rec = dict(i=i, psnr_TL_src=p_tl, psnr_resplat_BR=p_br, hole_dep=float(hole_dep.mean()),
               hole_me=float(hole_me.mean()), hole_disagree=float((hole_dep != hole_me).mean()),
               inferno_resid_med=float(np.median(np.sqrt(kd))), k_min=float(k.min()), k_max=float(k.max()))
    stats.append(rec)
    print(json.dumps(rec), file=log, flush=True)
    if i % 20 == 0:
        print(f"[{clip}] {i}/{T} {time.time() - t_start:.0f}s TLsrc {p_tl:.2f} resplat/BR {p_br:.2f} "
              f"holes dep {hole_dep.mean()*100:.2f}% me {hole_me.mean()*100:.2f}%", flush=True)
for a in (A_left, A_warp, A_mask, A_lrow, A_dep):
    a.flush()
meta = dict(clip=clip, splat_path=splat_path, left_source=leftsrc, T=T, fps=fps, Hq=Hq, Wq=Wq, top=top, lft=lft,
            depth_rows=[R0, R1], depthcrafter_proc=list(S.depthcrafter_proc_size(Hq, Wq)), lut=S.LUT_RT,
            median_psnr_TL_src=float(np.median([r["psnr_TL_src"] for r in stats])),
            min_psnr_TL_src=float(np.min([r["psnr_TL_src"] for r in stats])),
            median_psnr_resplat_BR=float(np.median([r["psnr_resplat_BR"] for r in stats])),
            mean_hole_disagree=float(np.mean([r["hole_disagree"] for r in stats])),
            mean_hole_dep=float(np.mean([r["hole_dep"] for r in stats])),
            mean_hole_me=float(np.mean([r["hole_me"] for r in stats])),
            seconds=time.time() - t_start)
json.dump(meta, open(os.path.join(od, "meta.json"), "w"), indent=1)
print(f"[{clip}] DONE {json.dumps(meta)}", flush=True)
