#!/usr/bin/env python
"""CPU-only: cost of main()'s end-of-clip stages on a REAL output array (no GPU).

Input: an existing FFV1 sbs render (decoded bit-exactly by decord).  Reproduces main()'s tail from the float32 frames:
  frames_output (T,3,H,W) float32 = u8/255.0 (what the PIL path yields), frames_left likewise
  A  sbs assembly: torch.cat([left, right], dim=3); (x*255).permute(0,2,3,1).to(uint8).cpu().numpy()
  B  anaglyph assembly: vid_left/vid_right uint8 conversions, channel zeroing, sum
  C  md5 of sbs / anaglyph (the lossless harness's writer_md5.txt; NOT part of the deployed CLI)
  D  FFV1 encode of sbs exactly as infer_lossless._ffv1_write (ffmpeg -c:v ffv1 -level 3 -g 1 -slicecrc 1 -threads 8)
  E  deployed writer: cv2 mp4v of sbs AND of the anaglyph (utils.inpainting.write_video_opencv), as the shipped CLI
  F  a streaming FFV1 writer fed window by window (identical bytes -> identical md5) is NOT timed here (needs the GPU
     loop); this script only reports the encode throughput so the overlap potential can be bounded.
All encoded files go to a NEW directory (argument 2); nothing existing is touched.
usage: write_bench_v1.py <sbs.mkv> <new_out_dir>
"""
import hashlib, os, subprocess, sys, time
import numpy as np
import torch

REPO = "/home/kawa/master_project/StereoCrafter"
sys.path.insert(0, REPO)
os.chdir(REPO)
from decord import VideoReader, cpu
from utils.inpainting import write_video_opencv as mp4v_write

src, outd = sys.argv[1], sys.argv[2]
assert not os.path.exists(outd), f"refusing to reuse {outd}"
os.makedirs(outd)
FFMPEG = "/home/kawa/miniconda3/envs/stereocrafter/bin/ffmpeg"
R = {}


def tick(name, fn):
    t0 = time.perf_counter()
    out = fn()
    R[name] = time.perf_counter() - t0
    return out


vr = VideoReader(src, ctx=cpu(0))
fps = float(vr.get_avg_fps())
sbs_u8 = tick("decode_input_for_bench", lambda: vr.get_batch(list(range(len(vr)))).asnumpy())
T, H, W2, _ = sbs_u8.shape
W = W2 // 2
R["shape"] = [T, H, W2, 3]
md5_in = hashlib.md5(np.ascontiguousarray(sbs_u8).tobytes()).hexdigest()
left = torch.from_numpy(sbs_u8[:, :, :W]).permute(0, 3, 1, 2).float() / 255.0
right = torch.from_numpy(sbs_u8[:, :, W:]).permute(0, 3, 1, 2).float() / 255.0
right = right.contiguous()


def assemble():
    frames_sbs = torch.cat([left, right], dim=3)
    return (frames_sbs * 255).permute(0, 2, 3, 1).to(dtype=torch.uint8).cpu().numpy()


fs = tick("A_sbs_assembly", assemble)
R["A_roundtrip_md5_equal_input"] = hashlib.md5(np.ascontiguousarray(fs).tobytes()).hexdigest() == md5_in


def anaglyph():
    vl = (left * 255).permute(0, 2, 3, 1).to(dtype=torch.uint8).cpu().numpy()
    vr_ = (right * 255).permute(0, 2, 3, 1).to(dtype=torch.uint8).cpu().numpy()
    vl[:, :, :, 1] = 0
    vl[:, :, :, 2] = 0
    vr_[:, :, :, 0] = 0
    return vl + vr_


ana = tick("B_anaglyph_assembly", anaglyph)
tick("C_md5_sbs", lambda: hashlib.md5(np.ascontiguousarray(fs).tobytes()).hexdigest())
tick("C_md5_anaglyph", lambda: hashlib.md5(np.ascontiguousarray(ana).tobytes()).hexdigest())


def ffv1(arr, path, threads=8):
    cmd = [FFMPEG, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{arr.shape[2]}x{arr.shape[1]}",
           "-r", f"{fps:.6f}", "-i", "-", "-an", "-c:v", "ffv1", "-level", "3", "-g", "1", "-slicecrc", "1",
           "-threads", str(threads), "-pix_fmt", "bgr0", path]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(arr.shape[0]):
        p.stdin.write(np.ascontiguousarray(arr[i]).tobytes())
    p.stdin.close()
    assert p.wait() == 0


tick("D_ffv1_encode_sbs_threads8", lambda: ffv1(fs, os.path.join(outd, "sbs_ffv1_t8.mkv"), 8))
tick("E_mp4v_sbs", lambda: mp4v_write(fs, fps, os.path.join(outd, "sbs_mp4v.mp4")))
tick("E_mp4v_anaglyph", lambda: mp4v_write(ana, fps, os.path.join(outd, "anaglyph_mp4v.mp4")))
R["load1"] = open("/proc/loadavg").read().split()[0]
R["torch_threads"] = torch.get_num_threads()
R["src"] = src
import json
json.dump(R, open(os.path.join(outd, "write_bench.json"), "w"), indent=1)
print(json.dumps(R, indent=1))
