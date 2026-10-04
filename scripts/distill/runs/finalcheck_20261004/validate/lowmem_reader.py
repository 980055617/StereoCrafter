"""Memory-light drop-in for utils.inpainting.read_and_prepare_video (return_right=False path only).

The tracked reader turns the WHOLE 2x2 splatting video into one float32 tensor (4400x4400 clips: ~36 GB, ~60-80 GB
peak with the decode copies).  This version decodes in chunks of CH frames and fills preallocated outputs with the
SAME element-wise ops on the same layouts:
  per chunk  f = torch.tensor(vr.get_batch(idx).asnumpy()).permute(0, 3, 1, 2).float()      (as the original)
             left   <- f[:, :, :H, :W][:, :, :H128, :W128] / 255.0
             warped <- f[:, :, H:, W:][:, :, :H128, :W128] / 255.0
             mask   <- (f[:, :, H:, :W][:, :, :H128, :W128] / 255.0).mean(dim=1, keepdim=True)
  outputs    left/warped dense channels-last (T,3,H128,W128), mask contiguous (T,1,H128,W128) -- the exact strides the
             original's ops produce (checked on CPU by test_lowmem_reader.py, end-to-end by the H1 md5 smoke).
Peak: the three outputs + one chunk (0042 at full length: ~21 GB instead of ~60-80 GB)."""
import torch
from decord import VideoReader, cpu

CH = 16


def read_and_prepare_video_lowmem(input_video_path: str, return_right: bool = False, _reader=VideoReader):
    if return_right:
        raise NotImplementedError("lowmem reader: return_right=True is not used by inpainting_inference.main")
    probe = _reader(input_video_path, ctx=cpu(0))
    H2, W2 = probe[0].shape[0], probe[0].shape[1]
    del probe
    video_reader = _reader(input_video_path, ctx=cpu(0))          # fresh reader: decode starts at frame 0 as before
    fps = float(video_reader.get_avg_fps())
    n = len(video_reader)
    height, width = H2 // 2, W2 // 2
    h128, w128 = height // 128 * 128, width // 128 * 128
    left = torch.empty_strided((n, 3, h128, w128), (h128 * w128 * 3, 1, w128 * 3, 3), dtype=torch.float32)
    warped = torch.empty_strided((n, 3, h128, w128), (h128 * w128 * 3, 1, w128 * 3, 3), dtype=torch.float32)
    mask = torch.empty_strided((n, 1, h128, w128), (h128 * w128, h128 * w128, w128, 1), dtype=torch.float32)
    for s in range(0, n, CH):
        idx = list(range(s, min(n, s + CH)))
        f = torch.tensor(video_reader.get_batch(idx).asnumpy()).permute(0, 3, 1, 2).float()
        e = s + len(idx)
        left[s:e] = f[:, :, :height, :width][:, :, :h128, :w128] / 255.0
        warped[s:e] = f[:, :, height:, width:][:, :, :h128, :w128] / 255.0
        mask[s:e] = (f[:, :, height:, :width][:, :, :h128, :w128] / 255.0).mean(dim=1, keepdim=True)
        del f
    return fps, left, warped, mask
