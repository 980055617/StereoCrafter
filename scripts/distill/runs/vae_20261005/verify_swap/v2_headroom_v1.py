#!/usr/bin/env python
"""vae_20261005 / verify_swap -- V2 (PREREG.txt): independent headroom re-score (decoder round trip of the REAL right eye).
Reference = decoder_ft gt_dev/<clip>_TR.npy (the frames that were encoded; registration-free).  Rows = decoder_swap's
headroom_dev/<clip>__<dec>.mkv; each mkv's full uint8 array md5 is checked against its .json md5.  Frames 0,4,8,...
LPIPS-Alex (batches of 4, sum / n), PSNR (mean of per-frame), reviewlib.decompose with regions from the GT frames.
usage: CUDA_VISIBLE_DEVICES=<g> flock /tmp/claude-gpu<g>.lock python v2_headroom_v1.py <out_json> <clip>
"""
import hashlib
import json
import math
import os
import sys
import time

import numpy as np
import torch
from decord import VideoReader, cpu

REPO = "/home/kawa/master_project/StereoCrafter"
os.chdir(REPO)
sys.path.insert(0, os.path.join(REPO, "scripts/distill/runs/review_20261001"))
import reviewlib as RL  # noqa: E402
import lpips  # noqa: E402

OUT, CLIP = sys.argv[1], sys.argv[2]
assert not os.path.exists(OUT), f"refusing to overwrite {OUT}"
HR = "/mnt/ssd_data/vae_20261005/decoder_swap/headroom_dev"
LANE = f"{REPO}/outputs/vae_20261005/decoder_swap/headroom_dev_v1/{CLIP}.json"
ROWS = ["stock", "ftmse", "ftema", "cd"]
dev = torch.device("cuda")
T0 = time.time()
X = np.load(f"/mnt/ssd_data/deep_20261004/decoder_ft/gt_dev/{CLIP}_TR.npy", mmap_mode="r")
n = X.shape[0]
fr = list(range(0, n, 4))
GT = np.ascontiguousarray(X[fr])
greg = RL.gray(GT)
reg = RL.regions(greg)
gdec = RL.decompose(reg, greg)
net = lpips.LPIPS(net="alex", verbose=False).to(dev).eval()
lane = json.load(open(LANE))


def t01(a):
    return torch.from_numpy(np.ascontiguousarray(a)).permute(0, 3, 1, 2).float() / 255.


res = dict(clip=CLIP, n=n, frames=fr, GT_decompose=gdec, rows={})
G = t01(GT)
with torch.no_grad():
    for r in ROWS:
        p = f"{HR}/{CLIP}__{r}.mkv"
        meta = json.load(open(f"{HR}/{CLIP}__{r}.json"))
        vr = VideoReader(p, ctx=cpu(0))
        assert len(vr) == n, (p, len(vr), n)
        allf = vr.get_batch(list(range(n))).asnumpy()
        md5 = hashlib.md5(np.ascontiguousarray(allf).tobytes()).hexdigest()
        Y = np.ascontiguousarray(allf[fr])
        del allf
        A = t01(Y)
        tot = 0.0
        for s in range(0, len(fr), 4):
            tot += float(net(A[s:s + 4].to(dev) * 2 - 1, G[s:s + 4].to(dev) * 2 - 1).sum())
        mse = [float(((A[j] - G[j]) ** 2).mean()) for j in range(len(fr))]
        d = RL.decompose(reg, RL.gray(Y))
        e = dict(path=p, md5=md5, md5_json=meta["md5"], md5_ok=md5 == meta["md5"], lpips=tot / len(fr),
                 psnr=float(np.mean([10 * math.log10(1 / max(m, 1e-12)) for m in mse])), decompose=d,
                 ratio_to_GT={k: (d[k] / gdec[k] if gdec[k] > 0 else float("nan")) for k in ("flatHF", "edgeHF", "stripeE")})
        lr = lane["rows"][r]
        e["lane"] = dict(lpips=lr["lpips"], psnr=lr["psnr"], ratio_to_GT=lr["ratio_to_GT"])
        e["diff"] = dict(lpips=e["lpips"] - lr["lpips"], psnr=e["psnr"] - lr["psnr"],
                         flatHF_rel=(d["flatHF"] - lr["decompose"]["flatHF"]) / lr["decompose"]["flatHF"],
                         edgeHF_rel=(d["edgeHF"] - lr["decompose"]["edgeHF"]) / lr["decompose"]["edgeHF"])
        res["rows"][r] = e
        print(f"[v2 {CLIP} {time.time() - T0:6.1f}s] {r:6s} md5_ok {e['md5_ok']} LPIPS {e['lpips']:.6f} (lane {lr['lpips']:.6f}, d "
              f"{e['diff']['lpips']:+.1e}) PSNR {e['psnr']:.4f} (d {e['diff']['psnr']:+.1e}) flatHF/GT {e['ratio_to_GT']['flatHF']:.3f} "
              f"edgeHF/GT {e['ratio_to_GT']['edgeHF']:.3f} rel {max(abs(e['diff']['flatHF_rel']), abs(e['diff']['edgeHF_rel'])):.1e}",
              flush=True)
res["seconds"] = time.time() - T0
json.dump(res, open(OUT, "w"), indent=1)
print("V2_DONE", CLIP, flush=True)
