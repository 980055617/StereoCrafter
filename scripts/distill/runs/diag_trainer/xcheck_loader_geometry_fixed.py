"""P0 re-check for fix F3: the trainer's REAL loader (utils/training_batches._StreamingVideo.load_chunk) vs the true-half quadrant
split of utils/inpainting.py:147-158.  The original xcheck_loader_geometry.py re-implemented the loader rule in numpy; this one
imports the loader so it measures the code that trains.  Pass = unshifted MAD 0.00 for cond/target/mask, mask top rows != left eye."""
import os, sys
REPO = "/home/kawa/master_project/StereoCrafter"; os.chdir(REPO); sys.path.insert(0, REPO)
import numpy as np, torch
from decord import VideoReader, cpu
from utils.training_batches import _StreamingVideo
def mad(a, b): return float(np.abs(np.asarray(a, np.float32) - np.asarray(b, np.float32)).mean())
nan = float("nan")
for clip in ["0154", "0011", "0358"]:
    path = f"video_data/train_gt28/{clip}_train.mp4"
    vr = VideoReader(path, ctx=cpu(0)); f = vr[30].asnumpy().astype(np.float32)
    H, W = f.shape[0] // 2, f.shape[1] // 2; th, tw = (H // 128) * 128, (W // 128) * 128; dh, dw = H - th, W - tw
    tTL = f[:H, :W]; tTR = f[:H, W:2 * W]; tBL = f[H:2 * H, :W].mean(axis=2); tBR = f[H:2 * H, W:2 * W]
    sv = _StreamingVideo(path); assert sv.spatial_hw == (th, tw), sv.spatial_hw
    cond, mask, target = sv.load_chunk(30, 31)
    lBR = cond[0].permute(1, 2, 0).numpy() * 255.0; lTR = target[0].permute(1, 2, 0).numpy() * 255.0; lBL = mask[0, 0].numpy() * 255.0
    print(clip, f.shape[:2], "tile", (th, tw), "offset (dh,dw)=", (dh, dw), "loader shapes", tuple(cond.shape), tuple(mask.shape), tuple(target.shape))
    print(f"  loader-target vs trueTR : unshifted {mad(lTR, tTR[:th, :tw]):.2f} | col-shifted by dw {mad(lTR[:, dw:], tTR[:th, :tw - dw]) if dw > 0 else nan:.2f}")
    print(f"  loader-cond   vs trueBR : unshifted {mad(lBR, tBR[:th, :tw]):.2f} | shifted (dh,dw) {mad(lBR[dh:, dw:], tBR[:th - dh, :tw - dw]) if (dh > 0 or dw > 0) else nan:.2f}")
    print(f"  loader-mask   vs trueBL : unshifted {mad(lBL, tBL[:th, :tw]):.2f} | row-shifted by dh {mad(lBL[dh:, :], tBL[:th - dh, :tw]) if dh > 0 else nan:.2f}")
    print(f"  loader-mask top {dh} rows: vs left-eye(TL) bottom rows {mad(lBL[:dh], tTL[H - dh:H, :tw].mean(axis=2)) if dh > 0 else nan:.2f} | vs trueBL top rows {mad(lBL[:dh], tBL[:dh, :tw]) if dh > 0 else nan:.2f}"
          f"  -> relative target-vs-cond displacement = {dh if mad(lBR, tBR[:th, :tw]) > 0.5 else 0} rows")
