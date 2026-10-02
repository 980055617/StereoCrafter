#!/usr/bin/env python
"""Run the tracked inpainting_inference with a LOSSLESS writer for the _sbs output.

Nothing in the tracked repo is modified. We import `inpainting_inference` and rebind the
name `write_video_opencv` inside that module's namespace before Fire dispatches main(),
so the diffusion/sampling path is byte-for-byte the shipped one.

Env knobs (all read at import):
  LOSSLESS_SBS=1        write <name>_sbs.mkv  (FFV1 level 3, pix_fmt bgr0, -g 1) instead of mp4v
  LOSSLESS_SBS=0        keep the original mp4v writer  (paired-control mode)
  KEEP_ANAGLYPH=0       default: skip the anaglyph file (diagnostic only, saves time+disk)
Always: append "<md5> <shape> <basename>" for every array handed to the writer to
  <save_dir>/writer_md5.txt   -- this is the pre-encode fingerprint used to prove that the
  lossless run and the mp4v control produced the identical pixel array.
"""
import hashlib
import os
import subprocess
import sys

import numpy as np

REPO = "/home/kawa/master_project/StereoCrafter"
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

FFMPEG = os.environ.get("FFMPEG_BIN", "/home/kawa/miniconda3/envs/stereocrafter/bin/ffmpeg")
LOSSLESS_SBS = os.environ.get("LOSSLESS_SBS", "1") not in ("0", "", "false", "False")
KEEP_ANAGLYPH = os.environ.get("KEEP_ANAGLYPH", "0") not in ("0", "", "false", "False")

import inpainting_inference as II  # noqa: E402  (after sys.path/chdir)
from utils.inpainting import write_video_opencv as _orig_write  # noqa: E402


def _ffv1_write(arr: np.ndarray, fps: float, path: str) -> None:
    """Write [T,H,W,3] uint8 RGB losslessly (FFV1 in Matroska, full-range RGB, all-keyframe)."""
    if arr.dtype != np.uint8:
        raise TypeError(f"lossless writer expects uint8, got {arr.dtype}")
    T, H, W, C = arr.shape
    if C != 3:
        raise ValueError(f"expected 3 channels, got {C}")
    cmd = [
        FFMPEG, "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", f"{float(fps):.6f}", "-i", "-",
        "-an", "-c:v", "ffv1", "-level", "3", "-g", "1", "-slicecrc", "1", "-threads", "8",
        "-pix_fmt", "bgr0", path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(T):
        proc.stdin.write(np.ascontiguousarray(arr[i]).tobytes())
    proc.stdin.close()
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"ffmpeg FFV1 encode failed rc={rc}: {' '.join(cmd)}")


def _patched_write(input_frames, fps, output_video_path):
    arr = np.ascontiguousarray(input_frames)
    base = os.path.basename(output_video_path)
    save_dir = os.path.dirname(output_video_path)
    digest = hashlib.md5(arr.tobytes()).hexdigest()
    with open(os.path.join(save_dir, "writer_md5.txt"), "a") as fh:
        fh.write(f"{digest} {tuple(arr.shape)} {arr.dtype} {base}\n")

    is_sbs = "_sbs" in base
    if not is_sbs and not KEEP_ANAGLYPH:
        print(f"[lossless] skipping anaglyph {base}")
        return
    if is_sbs and LOSSLESS_SBS:
        out = output_video_path[:-4] + ".mkv" if output_video_path.endswith(".mp4") else output_video_path + ".mkv"
        _ffv1_write(arr, fps, out)
        with open(out + ".md5", "w") as fh:
            fh.write(f"{digest}  {tuple(arr.shape)}  fps={float(fps):.6f}\n")
        print(f"[lossless] wrote FFV1 {out} md5(pre-encode)={digest}")
        return
    _orig_write(input_frames, fps, output_video_path)
    print(f"[lossless] wrote mp4v(original) {output_video_path} md5(pre-encode)={digest}")


II.write_video_opencv = _patched_write

if __name__ == "__main__":
    from fire import Fire
    print(f"[lossless] LOSSLESS_SBS={int(LOSSLESS_SBS)} KEEP_ANAGLYPH={int(KEEP_ANAGLYPH)} ffmpeg={FFMPEG}")
    Fire(II.run)
