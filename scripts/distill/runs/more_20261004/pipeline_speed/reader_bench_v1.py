#!/usr/bin/env python
"""CPU-only: tracked reader + main()'s center crop  vs  reader_cropfirst.read_cropfirst  (time, equality, strides).
usage: reader_bench_v1.py <clip> <H> <W> <out_json (new)>"""
import json, os, sys, time, hashlib
import torch

REPO = "/home/kawa/master_project/StereoCrafter"
sys.path.insert(0, REPO)
sys.path.insert(0, f"{REPO}/scripts/distill/runs/more_20261004/pipeline_speed")
os.chdir(REPO)
from utils.inpainting import read_and_prepare_video
from reader_cropfirst import read_cropfirst

clip, H, W, outj = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
assert not os.path.exists(outj), f"refusing to overwrite {outj}"
path = f"video_data/splatting/{clip}_splatting_results.mp4"


def center_crop(frames, crop_h, crop_w):            # == inpainting_inference._center_crop_frames
    h, w = int(frames.shape[2]), int(frames.shape[3])
    top, left = (h - crop_h) // 2, (w - crop_w) // 2
    return frames[:, :, top: top + crop_h, left: left + crop_w]


R = dict(clip=clip, H=H, W=W, load1_start=open("/proc/loadavg").read().split()[0], threads=torch.get_num_threads())
t0 = time.perf_counter()
fps_a, l_a, w_a, m_a = read_and_prepare_video(path)
t1 = time.perf_counter()
l_a, w_a, m_a = center_crop(l_a, H, W), center_crop(w_a, H, W), center_crop(m_a, H, W)
t2 = time.perf_counter()
R["tracked_read_s"], R["tracked_crop_s"] = t1 - t0, t2 - t1
t3 = time.perf_counter()
fps_b, l_b, w_b, m_b = read_cropfirst(path, H, W)
t4 = time.perf_counter()
l_b2, w_b2, m_b2 = center_crop(l_b, H, W), center_crop(w_b, H, W), center_crop(m_b, H, W)   # main() still calls it
R["cropfirst_read_s"] = t4 - t3
R["fps_equal"] = fps_a == fps_b
for n, a, b in (("left", l_a, l_b2), ("warped", w_a, w_b2), ("mask", m_a, m_b2)):
    R[f"{n}_equal"] = bool(torch.equal(a, b))
    R[f"{n}_shape"] = [list(a.shape), list(b.shape)]
    R[f"{n}_strides"] = [list(a.stride()), list(b.stride())]
    # what main() does with each: warped -> [cur_i:cur_i+14].clone(); mask -> [cur_i:cur_i+14] (view); left -> [:T]
    ca, cb = a[11:25].clone(), b[11:25].clone()
    R[f"{n}_window_clone_strides"] = [list(ca.stride()), list(cb.stride())]
    R[f"{n}_window_clone_equal"] = bool(torch.equal(ca, cb))
    R[f"{n}_md5"] = hashlib.md5(a.contiguous().numpy().tobytes()).hexdigest()
R["load1_end"] = open("/proc/loadavg").read().split()[0]
json.dump(R, open(outj, "w"), indent=1)
print(json.dumps(R, indent=1))
