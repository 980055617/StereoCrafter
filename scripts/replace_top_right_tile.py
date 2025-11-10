import os
from typing import Optional

import cv2
import numpy as np
from decord import VideoReader, cpu
from fire import Fire


def _read_video(path: str):
    vr = VideoReader(path, ctx=cpu(0))
    fps = float(vr.get_avg_fps())
    idx = list(range(len(vr)))
    frames = vr.get_batch(idx).asnumpy()  # (T, H, W, 3), uint8, RGB
    return fps, frames


def replace_top_right(
    input_2x2_video: str,
    right_video: str,
    output_video: str,
    *,
    resize_interpolation: str = "area",
    trim_to_min_frames: bool = True,
) -> None:
    """Replace the top-right tile of a 2x2 tiled video with frames from a right-view video.

    Assumptions
    - `input_2x2_video` is a 2x2 tiled RGB video. Top-left/Bottom-left/Bottom-right tiles are valid,
      and top-right will be overwritten.
    - `right_video` is a single-view RGB video containing the true right frames.
    - Both videos share (ideally) the same number of frames and fps. If frame counts differ and
      `trim_to_min_frames=True`, the output length is the minimum of the two.

    Args:
        input_2x2_video: Path to the source tiled video.
        right_video: Path to the right-view video to insert into the top-right tile.
        output_video: Path to write the resulting 2x2 tiled video (e.g., .mp4).
        resize_interpolation: Interpolation to use when resizing right-view to tile size. One of
            {"area", "linear", "cubic", "nearest"}. Default: "area".
        trim_to_min_frames: If True, trims to min(T_source, T_right). If False and lengths mismatch,
            raises an error.
    """
    if not os.path.exists(input_2x2_video):
        raise FileNotFoundError(f"input_2x2_video not found: {input_2x2_video}")
    if not os.path.exists(right_video):
        raise FileNotFoundError(f"right_video not found: {right_video}")

    fps_src, frames_src = _read_video(input_2x2_video)
    fps_right, frames_right = _read_video(right_video)

    T_src, H, W, C = frames_src.shape
    if C != 3:
        raise ValueError(f"Expected 3 channels for source, got {C}")
    if H % 2 != 0 or W % 2 != 0:
        raise ValueError(f"Source frame size must be even divisible (H={H}, W={W}) for 2x2 tiling")

    T_right, Hr, Wr, Cr = frames_right.shape
    if Cr != 3:
        raise ValueError(f"Expected 3 channels for right, got {Cr}")

    # Length alignment
    if T_src != T_right:
        if trim_to_min_frames:
            T_out = min(T_src, T_right)
            frames_src = frames_src[:T_out]
            frames_right = frames_right[:T_out]
        else:
            raise ValueError(f"Frame count mismatch: source={T_src}, right={T_right}")
    else:
        T_out = T_src

    # Determine tile sizes from source
    tile_h, tile_w = H // 2, W // 2

    # Choose interpolation
    interp_map = {
        "area": cv2.INTER_AREA,
        "linear": cv2.INTER_LINEAR,
        "nearest": cv2.INTER_NEAREST,
        "cubic": cv2.INTER_CUBIC,
    }
    if resize_interpolation not in interp_map:
        raise ValueError(f"Unknown resize_interpolation: {resize_interpolation}")
    interp_flag = interp_map[resize_interpolation]

    out_frames = np.empty((T_out, H, W, 3), dtype=np.uint8)

    for i in range(T_out):
        src = frames_src[i].copy()  # RGB
        rgt = frames_right[i]

        # Resize right to tile size if needed
        if rgt.shape[0] != tile_h or rgt.shape[1] != tile_w:
            # cv2.resize expects (width, height)
            rgt = cv2.resize(rgt, (tile_w, tile_h), interpolation=interp_flag)

        # Place right view into top-right tile
        src[0:tile_h, tile_w:W, :] = rgt
        out_frames[i] = src

    # Use source fps to keep layout timing consistent
    from utils.inpainting import write_video_opencv

    _, output_ext = os.path.splitext(output_video)
    if output_ext == "":
        output_video = f"{output_video}.mp4"

    os.makedirs(os.path.dirname(os.path.abspath(output_video)), exist_ok=True)
    write_video_opencv(out_frames, fps_src, output_video)

    # Minimal console feedback
    print(
        f"Wrote: {output_video} | frames={T_out} | fps={fps_src:.2f} | size={W}x{H} | top-right replaced"
    )


def main(
    input_2x2_video: str,
    right_video: str,
    output_video: str,
    resize_interpolation: str = "area",
    trim_to_min_frames: bool = True,
):
    replace_top_right(
        input_2x2_video,
        right_video,
        output_video,
        resize_interpolation=resize_interpolation,
        trim_to_min_frames=trim_to_min_frames,
    )


if __name__ == "__main__":
    Fire(main)
