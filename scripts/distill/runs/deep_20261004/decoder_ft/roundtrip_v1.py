#!/usr/bin/env python
"""deep_20261004 / decoder_ft lane -- DECODER-ONLY round trip of the REAL right eye (diagnostic; never used for selection).

GT frames -> deployed encode (image_processor.preprocess -> _encode_vae_frames(n_frames_per_time=5), mode) per deployed
14/3 window -> (z*sf).bf16 -> decode_latents(decode_chunk_size=2) with each decoder -> tensor2vid(pil) -> keep rule ->
uint8.  The latents are computed ONCE per clip and shared by every decoder (the encoder is not bit-deterministic across
calls; sharing makes the decoder contrast exact).  Metrics on frames 0,4,8,... (score_clip_ll grid) vs the input frames
themselves (registration-free): LPIPS-Alex (batches of 4, score_clip_ll call), PSNR, DISTS.
Sources: dev  /mnt/ssd_data/deep_20261004/decoder_ft/gt_dev/<clip>_TR.npy (extract_gt_v1.py, unregistered deployed window)
         test /mnt/ssd_data/deep_20261004/blur_diag/cache_r2/<clip>/GTreg.npy (blur_diag's VAE_GT input; read-only)
usage: CUDA_VISIBLE_DEVICES=1 flock /tmp/claude-gpu1.lock python roundtrip_v1.py <out_json> <dev|test> <clips> <dec_spec>
  dec_spec as redecode_v1.py ("stock" is always included first)
"""
import hashlib
import json
import math
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

REPO = "/home/kawa/master_project/StereoCrafter"
sys.path.insert(0, REPO)
os.chdir(REPO)
os.environ.setdefault("TORCH_HOME", "/mnt/ssd_data/deep_20261004/decoder_ft/torch_home")
import inpainting_inference as II  # noqa: E402
import lpips  # noqa: E402
import piq  # noqa: E402
from diffusers.image_processor import VaeImageProcessor  # noqa: E402
from diffusers.models.autoencoders.autoencoder_kl_temporal_decoder import AutoencoderKLTemporalDecoder  # noqa: E402

PRE = "weights/stable-video-diffusion-img2vid-xt-1-1/"
FC, OV, DCS = 14, 3, 2
dt = torch.bfloat16
OUTJ, KIND, CLIPS, DSPEC = sys.argv[1], sys.argv[2], sys.argv[3].split(","), sys.argv[4]
assert not os.path.exists(OUTJ), f"refusing to overwrite {OUTJ}"
T0 = time.time()


def log(*a):
    print(f"[rt {time.time() - T0:7.1f}s]", *a, flush=True)


def src(clip):
    if KIND == "dev":
        return np.load(f"/mnt/ssd_data/deep_20261004/decoder_ft/gt_dev/{clip}_TR.npy", mmap_mode="r")
    return np.load(f"/mnt/ssd_data/deep_20261004/blur_diag/cache_r2/{clip}/GTreg.npy", mmap_mode="r")


def windows(n):
    out, gen = [], False
    for i in range(0, n, FC - OV):
        if i + OV >= n:
            break
        if gen and i + FC > n:
            cur_i = max(n + OV - FC, 0)
            cur_ov = i - cur_i + OV
        else:
            cur_i, cur_ov = i, OV
        out.append((i, cur_i, cur_ov))
        gen = True
    return out


vae = AutoencoderKLTemporalDecoder.from_pretrained(PRE, subfolder="vae", variant="fp16", torch_dtype=dt)
vae.requires_grad_(False)
vae.to(dtype=dt)
vae = vae.to("cuda").eval()
shim = SimpleNamespace(vae=vae, vae_scale_factor=8, image_processor=VaeImageProcessor(vae_scale_factor=8))
stock = {k: v.detach().clone() for k, v in vae.decoder.state_dict().items()}
decs = [("stock", None)] + ([] if DSPEC == "stock" else [tuple(x.split("=", 1)) for x in DSPEC.split(",")])
net = lpips.LPIPS(net="alex", verbose=False).cuda().eval()
dists = piq.DISTS(reduction="none").cuda()
res = dict(kind=KIND, clips=CLIPS, decoders=[d[0] for d in decs], per_clip={})
with torch.no_grad():
    for clip in CLIPS:
        X = src(clip)
        n = X.shape[0]
        W = windows(n)
        lats = []
        for (i, cur_i, cur_ov) in W:
            x = torch.from_numpy(np.ascontiguousarray(X[cur_i:cur_i + FC])).permute(0, 3, 1, 2).float() / 255.0
            xp = shim.image_processor.preprocess(x, height=x.shape[2], width=x.shape[3])
            z = II._Pipe._encode_vae_frames(shim, xp, torch.device("cuda"), 1, False, n_frames_per_time=5)
            lats.append((z.float() * vae.config.scaling_factor).to(dt))
        fr = list(range(0, n, 4))
        G = torch.from_numpy(np.ascontiguousarray(X[fr])).permute(0, 3, 1, 2).float() / 255.0
        pc = {}
        for name, path in decs:
            sd = stock if path is None else torch.load(path, map_location="cpu", weights_only=False)["decoder"]
            own = vae.decoder.state_dict()
            for k, p in own.items():
                p.copy_(sd[k].to(device=p.device, dtype=p.dtype))
            outs = []
            for (i, cur_i, cur_ov), lat in zip(W, lats):
                vf = II._Pipe.decode_latents(shim, lat, num_frames=lat.shape[1], decode_chunk_size=DCS)
                vf = II.tensor2vid(vf, shim.image_processor, output_type="pil")[0]
                g = torch.stack([torch.tensor(np.array(im)).permute(2, 0, 1).float() / 255.0 for im in vf])
                outs.append(g if i == 0 else g[cur_ov:])
            out = torch.cat(outs)
            assert out.shape[0] == n
            u8 = (out * 255).to(torch.uint8)
            Y = u8[fr].float() / 255.0
            tot, dl = 0.0, []
            for s in range(0, len(fr), 4):
                tot += float(net(Y[s:s + 4].cuda() * 2 - 1, G[s:s + 4].cuda() * 2 - 1).sum())
                dl += [float(v) for v in dists(Y[s:s + 4].cuda(), G[s:s + 4].cuda()).view(-1)]
            mse = [(Y[j] - G[j]).pow(2).mean().item() for j in range(len(fr))]
            pc[name] = dict(lpips=tot / len(fr), psnr=float(np.mean([10 * math.log10(1 / max(m, 1e-12)) for m in mse])),
                            dists=float(np.mean(dl)), md5=hashlib.md5(u8.numpy().tobytes()).hexdigest())
            log(f"{clip} {name:14s} LPIPS-alex {pc[name]['lpips']:.4f} PSNR {pc[name]['psnr']:.3f} DISTS {pc[name]['dists']:.4f}")
        res["per_clip"][clip] = pc
res["mean"] = {d[0]: {m: float(np.mean([res["per_clip"][c][d[0]][m] for c in CLIPS])) for m in ("lpips", "psnr", "dists")}
               for d in decs}
res["seconds"] = time.time() - T0
json.dump(res, open(OUTJ, "w"), indent=1)
for d in decs:
    m = res["mean"][d[0]]
    log(f"MEAN {d[0]:14s} LPIPS-alex {m['lpips']:.4f} PSNR {m['psnr']:.3f} DISTS {m['dists']:.4f}")
log("ROUNDTRIP_DONE")
