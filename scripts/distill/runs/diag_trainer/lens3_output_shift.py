"""Vertical/horizontal displacement of rendered right-eye outputs vs real GT (CPU). Uses the scorer's left-eye alignment."""
import os, sys, json, math
REPO = "/home/kawa/master_project/StereoCrafter"; os.chdir(REPO)
import numpy as np, torch, torch.nn.functional as F
from decord import VideoReader, cpu
STEP = 15
def load(p):
    vr = VideoReader(p, ctx=cpu(0)); idx = list(range(0, len(vr), STEP))
    return torch.from_numpy(vr.get_batch(idx).asnumpy()).permute(0, 3, 1, 2).float() / 255.
clips = sys.argv[1].split(","); tags = sys.argv[2].split(",")
OUT = {}
for clip in clips:
    tile = load(f"video_data/train/{clip}_train.mp4"); H, W = tile.shape[2] // 2, tile.shape[3] // 2
    gtL, gtR = tile[:, :, :H, :W], tile[:, :, :H, W:2 * W]
    R_ = {}
    for tag in tags:
        p = f"outputs/fulldata_v2/clips/{clip}_{tag}/{clip}_inpainting_results_sbs.mp4"
        if not os.path.exists(p): R_[tag] = "missing"; continue
        v = load(p); half = v.shape[3] // 2; L, R = v[:, :, :, :half], v[:, :, :, half:]
        n = min(len(L), len(gtL)); L, R = L[:n], R[:n]; h, w = L.shape[2], L.shape[3]
        # scorer-style alignment on the LEFT eye (coarse then fine)
        best = ((0, 0), 1e9)
        for st, rng in ((4, 60), (1, 6)):
            cy, cx = best[0]
            for dy in range(cy - rng, cy + rng + 1, st):
                for dx in range(cx - rng, cx + rng + 1, st):
                    t0 = (H - h) // 2 + dy; l0 = (W - w) // 2 + dx
                    if t0 < 0 or l0 < 0 or t0 + h > H or l0 + w > W: continue
                    m = (L - gtL[:n, :, t0:t0 + h, l0:l0 + w]).pow(2).mean().item()
                    if m < best[1]: best = ((dy, dx), m)
        (dy, dx), _ = best; t0 = (H - h) // 2 + dy; l0 = (W - w) // 2 + dx
        # now: displacement of the RIGHT output vs GT right, searched around the left alignment
        res = {}
        for ddy in range(-72, 73, 4):
            for ddx in range(-32, 33, 4):
                tt, ll = t0 + ddy, l0 + ddx
                if tt < 0 or ll < 0 or tt + h > H or ll + w > W: continue
                res[(ddy, ddx)] = (R - gtR[:n, :, tt:tt + h, ll:ll + w]).pow(2).mean().item()
        bkey = min(res, key=res.get)
        # refine
        for ddy in range(bkey[0] - 3, bkey[0] + 4):
            for ddx in range(bkey[1] - 3, bkey[1] + 4):
                tt, ll = t0 + ddy, l0 + ddx
                if tt < 0 or ll < 0 or tt + h > H or ll + w > W or (ddy, ddx) in res: continue
                res[(ddy, ddx)] = (R - gtR[:n, :, tt:tt + h, ll:ll + w]).pow(2).mean().item()
        bkey = min(res, key=res.get)
        # column-profile check: vertical-only shift using the 1-D row-mean profile (robust to disparity)
        prof = {}
        for ddy in range(-72, 73, 2):
            tt = t0 + ddy
            if tt < 0 or tt + h > H: continue
            prof[ddy] = (R.mean(dim=3) - gtR[:n, :, tt:tt + h, l0:l0 + w].mean(dim=3)).pow(2).mean().item()
        pbest = min(prof, key=prof.get)
        sharp = (R[:, :, :, 1:] - R[:, :, :, :-1]).abs().mean().item()
        R_[tag] = dict(left_align=(dy, dx), right_best_shift=bkey, psnr_at_0=round(10 * math.log10(1 / max(res[(0, 0)], 1e-12)), 2),
                       psnr_at_best=round(10 * math.log10(1 / max(res[bkey], 1e-12)), 2), rowprofile_best_dy=pbest,
                       rowprofile_mse_0_vs_best=(round(prof[0], 6), round(prof[pbest], 6)), sharp=round(sharp, 4), frames=n)
        print(clip, tag, R_[tag], flush=True)
    OUT[clip] = R_
json.dump(OUT, open("scripts/distill/runs/diag_trainer/lens3_output_shift.json", "w"), indent=1, default=str)
