"""Crop-first drop-in for main()'s  read_and_prepare_video(path) + _center_crop_frames(...) x3  (return_right=False).

The tracked reader converts the WHOLE 2x2 tile video to float32 (0301: 151x3x2160x3840 = 15 GB), divides all four
quadrants by 255 (incl. the unused right-eye quadrant) and only then center-crops (views).  This version slices the
uint8 frames to the final crop FIRST and applies the identical element-wise ops (float(), /255.0, mean over the 3
channels) to just that region.  Per-element values are identical by construction (element-wise ops; the mask's channel
mean sums the same 3 contiguous floats in the same order); the left/warped tensors come out channels-last dense
(T,3,h,w), which is what main()'s .clone() of the tracked center-cropped view produces anyway.  Verified tensor-equal
(torch.equal + .clone() strides) against the tracked path by reader_bench_v1.py before any use.
"""
import torch
from decord import VideoReader, cpu


def read_cropfirst(input_video_path: str, crop_h: int, crop_w: int):
    vr = VideoReader(input_video_path, ctx=cpu(0))
    fps = float(vr.get_avg_fps())
    u8 = torch.tensor(vr.get_batch(list(range(len(vr)))).asnumpy())      # (T, 2H, 2W, 3) uint8, as the original
    T, H2, W2, _ = u8.shape
    height, width = H2 // 2, W2 // 2
    h128, w128 = height // 128 * 128, width // 128 * 128
    if crop_h > h128 or crop_w > w128:
        raise ValueError(f"Requested crop {crop_h}x{crop_w} exceeds source {h128}x{w128}.")
    top, left = (h128 - crop_h) // 2, (w128 - crop_w) // 2

    def quad(r0, c0):
        v = u8[:, r0 + top: r0 + top + crop_h, c0 + left: c0 + left + crop_w, :]
        return v.permute(0, 3, 1, 2).float()                               # (T,3,h,w) channels-last dense float32

    frames_left = quad(0, 0) / 255.0
    frames_warped = quad(height, width) / 255.0
    frames_mask = (quad(height, 0) / 255.0).mean(dim=1, keepdim=True)
    return fps, frames_left, frames_warped, frames_mask
