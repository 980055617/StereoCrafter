from decord import VideoReader, cpu
from utils.training_batches import chunk_frame_ranges

vr = VideoReader("/workspace/stereocraft/video_data/train/0160_train.mp4", ctx=cpu(0))
num_frames = len(vr)
chunk = 23
overlap = 3

ranges = list(chunk_frame_ranges(num_frames, chunk, overlap))
print("ranges:", ranges)

for i in range(1, len(ranges)):
    prev_s, prev_e = ranges[i-1]
    s, e = ranges[i]
    actual_ov = max(0, prev_e - s)
    if actual_ov != overlap:
        print("WARNING: actual overlap != config overlap",
              "i=", i, "actual=", actual_ov, "config=", overlap, "prev=", (prev_s,prev_e), "cur=", (s,e))
