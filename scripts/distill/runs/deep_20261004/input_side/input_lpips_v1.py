"""DESCRIPTIVE (not pre-registered): LPIPS-Alex of the model INPUT (warped window, holes black) against the real right
eye, UNREG and REG_FRAME (each input's own registration from its score_input_v1.py JSON), scored frames.  CPU.
usage: input_lpips_v1.py <out_json> <clip> <input> [<input> ...]"""
import json, os, sys
import numpy as np, torch, lpips
from decord import VideoReader, cpu
REPO = "/home/kawa/master_project/StereoCrafter"; os.chdir(REPO)
torch.set_num_threads(8)
OUT, CLIP = sys.argv[1], sys.argv[2]
assert not os.path.exists(OUT), OUT
net = lpips.LPIPS(net="alex").eval()
res = {}
vt = VideoReader(f"video_data/train/{CLIP}_train.mp4", ctx=cpu(0))
for inp in sys.argv[3:]:
    j = json.load(open(f"outputs/deep_20261004/input_side/scores_v1/{CLIP}__{inp}.json"))
    fr = j["frames"]; t0, l0 = j["window"]; sy, sx = j["reg"]["smooth_ddy"], j["reg"]["smooth_ddx"]
    W8 = np.load(f"/mnt/ssd_data/deep_20261004/input_side/inputs_v1/{CLIP}/{inp}/warped.npy", mmap_mode="r")
    tot = {"UNREG": [], "REG_FRAME": []}
    for s in range(0, len(fr), 4):
        part = fr[s:s + 4]
        b = vt.get_batch(part).asnumpy(); H, W = b.shape[1] // 2, b.shape[2] // 2
        x = torch.from_numpy(np.stack([np.asarray(W8[f]) for f in part])).permute(0, 3, 1, 2).float() / 255.
        for var in tot:
            g = np.stack([b[k, t0 + (sy[f] if var != "UNREG" else 0):t0 + (sy[f] if var != "UNREG" else 0) + 576,
                            W + l0 + (sx[f] if var != "UNREG" else 0):W + l0 + (sx[f] if var != "UNREG" else 0) + 1024]
                          for k, f in enumerate(part)])
            y = torch.from_numpy(g).permute(0, 3, 1, 2).float() / 255.
            with torch.no_grad():
                tot[var] += [float(v) for v in net(x * 2 - 1, y * 2 - 1).view(-1)]
    res[inp] = {k: float(np.mean(v)) for k, v in tot.items()}
    print(CLIP, inp, {k: round(v, 4) for k, v in res[inp].items()}, flush=True)
json.dump(dict(clip=CLIP, note="DESCRIPTIVE input LPIPS vs real right eye", res=res), open(OUT, "w"), indent=1)
