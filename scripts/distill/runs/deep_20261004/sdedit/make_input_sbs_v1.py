#!/usr/bin/env python
"""deep_20261004 / sdedit lane: DESCRIPTIVE reference rows (CPU only; never gating).
Writes the model's own warped right-eye INPUT as if it were a render, so the unchanged scorers can score it:
  <OUT>/clips/<clip>_INPUT_warp/<clip>_inpainting_results_sbs.mkv   [left | warped input, holes as they are]
  <OUT>/clips/<clip>_INPUT_fill/<clip>_inpainting_results_sbs.mkv   [left | warped input, every hole pixel row-filled]
These are the sigma -> 0 limits of sd1 / sd1fill (no VAE round trip, no denoising).
Frames are produced exactly like inpainting_inference.main: decord uint8 -> float32 / 255 -> crop to /128 ->
_center_crop_frames(576, 1024); mask = RGB mean / 255; the sbs array is (float * 255).to(uint8) of [left | right]
(the renders' own conversion, so the left half is bit-identical to every render's left half; K3 checks it).
The fill is the hook's: sdfill_v1.fill_all_rowlin on the window + 16 px margin, hole = mask >= 0.5.
The writer is infer_lossless._patched_write (FFV1 + writer_md5.txt), imported by path.  Refuses existing dirs.
usage: CUDA_VISIBLE_DEVICES= python make_input_sbs_v1.py OUTROOT clip [clip ...]
"""
import importlib.util
import os
import sys

import numpy as np
import torch
from decord import VideoReader, cpu

REPO = "/home/kawa/master_project/StereoCrafter"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sdfill_v1 as SDF  # noqa: E402

os.chdir(REPO)
_spec = importlib.util.spec_from_file_location("infer_lossless", f"{REPO}/scripts/distill/runs/fulldata_v2/beyond4/infer_lossless.py")
_IL = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_IL)

TH, TW, M = 576, 1024, 16
OUTROOT = sys.argv[1]
for clip in sys.argv[2:]:
    dirs = {k: os.path.join(OUTROOT, "clips", f"{clip}_INPUT_{k}") for k in ("warp", "fill")}
    for d in dirs.values():
        if os.path.exists(d):
            sys.exit(f"refusing to overwrite {d}")
    vr = VideoReader(f"video_data/splatting/{clip}_splatting_results.mp4", ctx=cpu(0))
    fps = float(vr.get_avg_fps())
    T = len(vr)
    arr = {k: [] for k in dirs}
    nh = nc = 0
    for s in range(0, T, 8):
        b = vr.get_batch(list(range(s, min(T, s + 8)))).asnumpy()
        for f in b:
            fr = torch.from_numpy(f).permute(2, 0, 1).float()          # [3, 2H, 2W]
            H, W = fr.shape[1] // 2, fr.shape[2] // 2
            left, mask, warped = fr[:, :H, :W], fr[:, H:, :W], fr[:, H:, W:]
            h128, w128 = H // 128 * 128, W // 128 * 128
            left, mask, warped = left[:, :h128, :w128] / 255.0, (mask[:, :h128, :w128] / 255.0).mean(0), warped[:, :h128, :w128] / 255.0
            top, left0 = (h128 - TH) // 2, (w128 - TW) // 2
            lw = left[:, top:top + TH, left0:left0 + TW]
            ww = warped[:, top:top + TH, left0:left0 + TW]
            y0, y1, x0, x1 = max(0, top - M), min(h128, top + TH + M), max(0, left0 - M), min(w128, left0 + TW + M)
            reg = warped[:, y0:y1, x0:x1].permute(1, 2, 0).contiguous().numpy()
            holes = mask[y0:y1, x0:x1].numpy() >= 0.5
            filled, st = SDF.fill_all_rowlin(reg, holes)
            wy, wx = top - y0, left0 - x0
            wf = torch.from_numpy(np.ascontiguousarray(filled[wy:wy + TH, wx:wx + TW])).permute(2, 0, 1)
            nh += int(holes[wy:wy + TH, wx:wx + TW].sum())
            nc += int(st["changed"][wy:wy + TH, wx:wx + TW].sum())
            for k, rr in (("warp", ww), ("fill", wf)):
                sbs = torch.cat([lw, rr], dim=2)
                arr[k].append((sbs * 255).permute(1, 2, 0).to(dtype=torch.uint8).numpy())
    for k, d in dirs.items():
        os.makedirs(d)
        a = np.stack(arr[k])
        _IL._patched_write(a, fps, os.path.join(d, f"{clip}_inpainting_results_sbs.mp4"))
    print(f"{clip}: frames {T} fps {fps:.3f} window ({top},{left0}) hole px {nh} filled px {nc} -> {list(dirs.values())}",
          flush=True)
print("DONE")
