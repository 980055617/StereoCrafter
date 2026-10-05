#!/usr/bin/env python
"""blur_diag (deep_20261004) -- HIRES_B working-resolution render hook (ORIGIN only).  PREREG.txt rows HIRES_B, gate G3.

Deployed origin inference (config/0160_overfit_inference_matched.json, unet_state_path None, 8 steps, guidance 1.01,
seed 1234, tile_num 1) with the lossless FFV1 writer of scripts/distill/runs/fulldata_v2/beyond4/infer_lossless.py.
The ONLY change is the input reader:
  reader = finalcheck_20261004/validate/lowmem_reader.read_and_prepare_video_lowmem (proven bit-identical to the
           tracked reader, H1 md5 PASS), then
  IDENTITY mode (UP_H=576 UP_W=1024): returned unchanged -> main() centre-crops exactly as deployed (gate G3 control)
  UPSAMPLE mode: the deployed 576x1024 window (inpainting_inference._center_crop_frames, the function main() uses) of
           left / warped is bicubic-upsampled (align_corners=False, clamped to [0,1]) and the mask nearest-upsampled to
           UP_H x UP_W; main() is run with target UP_H x UP_W (its centre crop is then the identity).
  writer (UPSAMPLE mode only): the SBS array's right half is cv2.INTER_AREA-downsampled to 576x1024 and paired with the
           native left window computed with main()'s own ops ((left*255).to(uint8)); that 576x2048 array goes through the
           unchanged lossless writer (FFV1 + writer_md5.txt).  The full-resolution right eye is also kept losslessly as
           <name>_hires_right_<H>x<W>.mkv for inspection.
env: UP_CLIP, UP_OUT (new dir), UP_H, UP_W, UP_MAXCHUNKS (optional, smoke), LOSSLESS_SBS=1, KEEP_ANAGLYPH=0,
     MAMBA_SELF_ATTN_INCLUDE='__nomatch__' (origin, as every origin_ll render)
"""
import importlib.util
import os
import sys

REPO = "/home/kawa/master_project/StereoCrafter"
sys.path.insert(0, REPO)
os.chdir(REPO)
_spec = importlib.util.spec_from_file_location(
    "infer_lossless", f"{REPO}/scripts/distill/runs/fulldata_v2/beyond4/infer_lossless.py")
_IL = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_IL)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import inpainting_inference as ii  # noqa: E402

sys.path.insert(0, f"{REPO}/scripts/distill/runs/finalcheck_20261004/validate")
from lowmem_reader import read_and_prepare_video_lowmem as _lowmem_read  # noqa: E402

CLIP = os.environ["UP_CLIP"]
OUT = os.environ["UP_OUT"]
UH, UW = int(os.environ.get("UP_H", "1024")), int(os.environ.get("UP_W", "1792"))
MAXCH = os.environ.get("UP_MAXCHUNKS", "").strip()
NH, NW = 576, 1024
IDENT = (UH, UW) == (NH, NW)
assert ii.write_video_opencv is _IL._patched_write, "lossless writer patch did not take"
assert os.environ.get("MAMBA_SELF_ATTN_INCLUDE") == "__nomatch__", "origin must run with MAMBA_SELF_ATTN_INCLUDE=__nomatch__"
NATIVE = {}


def _up(x, mode):
    outs = []
    for s in range(0, x.shape[0], 8):
        if mode == "bicubic":
            outs.append(F.interpolate(x[s:s + 8], size=(UH, UW), mode="bicubic", align_corners=False).clamp_(0, 1))
        else:
            outs.append(F.interpolate(x[s:s + 8], size=(UH, UW), mode="nearest"))
    return torch.cat(outs).contiguous(memory_format=torch.channels_last) if mode == "bicubic" else torch.cat(outs)


def reader(input_video_path, return_right=False):
    fps, left, warped, mask = _lowmem_read(input_video_path, return_right=return_right)
    if IDENT:
        print(f"[up] IDENTITY mode: lowmem reader output unchanged {tuple(left.shape)}", flush=True)
        return fps, left, warped, mask
    lw = ii._center_crop_frames(left, NH, NW)
    ww = ii._center_crop_frames(warped, NH, NW)
    mw = ii._center_crop_frames(mask, NH, NW)
    NATIVE["left_u8"] = (lw * 255).permute(0, 2, 3, 1).to(dtype=torch.uint8).cpu().numpy()
    lu, wu, mu = _up(lw, "bicubic"), _up(ww, "bicubic"), _up(mw, "nearest")
    del left, warped, mask
    print(f"[up] UPSAMPLE mode: window {tuple(lw.shape)} -> {tuple(wu.shape)} (bicubic; mask nearest)", flush=True)
    return fps, lu, wu, mu


def writer(input_frames, fps, output_video_path):
    base = os.path.basename(output_video_path)
    if IDENT or "_sbs" not in base:
        return _IL._patched_write(input_frames, fps, output_video_path)
    arr = np.ascontiguousarray(input_frames)
    assert arr.shape[1:] == (UH, 2 * UW, 3), arr.shape
    right_big = np.ascontiguousarray(arr[:, :, UW:])
    _IL._ffv1_write(right_big, fps, os.path.join(os.path.dirname(output_video_path),
                                                 base.replace("_sbs.mp4", f"_hires_right_{UH}x{UW}.mkv")))
    right = np.stack([cv2.resize(f, (NW, NH), interpolation=cv2.INTER_AREA) for f in right_big])
    left = NATIVE["left_u8"][:len(right)]
    print(f"[up] writer: right {right_big.shape} -> {right.shape} INTER_AREA; native left {left.shape}", flush=True)
    return _IL._patched_write(np.concatenate([left, right], axis=2), fps, output_video_path)


ii.read_and_prepare_video = reader
ii.write_video_opencv = writer
print(f"[up] clip={CLIP} out={OUT} res={UH}x{UW} mode={'IDENTITY' if IDENT else 'UPSAMPLE'} maxchunks={MAXCH or None} "
      f"unet=None", flush=True)
kw = dict(config="config/0160_overfit_inference_matched.json",
          input_video_path=f"video_data/splatting/{CLIP}_splatting_results.mp4",
          save_dir=OUT, num_inference_steps=8, min_guidance_scale=1.01, max_guidance_scale=1.01,
          unet_state_path=None, target_height=UH, target_width=UW, tile_num=1)
if MAXCH:
    kw["max_profile_chunks"] = int(MAXCH)
ii.run(**kw)
print("[up] done", flush=True)
