#!/usr/bin/env python
"""deep_20261004 / decoder_ft lane -- encode the decoder's TRAINING latents (GPU, run under the GPU lock).

Source (read-only, lossless): scale_gt's registered real-right-eye crops
  /mnt/ssd_data/deep_20261004/scale_gt/cache_v1/crops/<clip>/w<start>/tgt.npy   uint8 [14,576,1024,3]
for the 291 GT-valid train clips (scripts/distill/runs/deep_20261004/scale_gt/clips_valid_v1.json summary.train_valid:
int < 0310, not a train_leftGT_broken link, real right eye != left eye; no test or dev clip).  The registration shift only
moves the crop window inside the real right eye; for an autoencoder target (latent of a frame -> that same frame) it is
irrelevant.  The md5 of every tgt.npy is re-checked against its clip.json before use.
Encode path = the deployed conditioning encode, verbatim: image_processor.preprocess(frames, 576, 1024) ->
  MambaStableVideoDiffusionInpaintingPipeline._encode_vae_frames(shim, x, cuda, 1, False, n_frames_per_time=5)
  (vae.encode(...).latent_dist.mode() in groups of 5) with the VAE loaded as in inpainting_inference.main (fp16 variant ->
  bf16).  Stored UNSCALED (the mode itself, bf16) as <out>/<clip>_w<start:03d>.pt = dict(mode=[14,4,72,128] bf16, meta).
  The trainer applies the deployed scaling round trip ((mode*sf).bf16 then decode_latents' 1/sf*) itself.
usage: CUDA_VISIBLE_DEVICES=1 flock /tmp/claude-gpu1.lock python encode_gt_v1.py <out_dir> [clip,clip,...]
"""
import hashlib
import json
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

REPO = "/home/kawa/master_project/StereoCrafter"
sys.path.insert(0, REPO)
os.chdir(REPO)
import inpainting_inference as II  # noqa: E402
from diffusers.image_processor import VaeImageProcessor  # noqa: E402
from diffusers.models.autoencoders.autoencoder_kl_temporal_decoder import AutoencoderKLTemporalDecoder  # noqa: E402

CROPS = "/mnt/ssd_data/deep_20261004/scale_gt/cache_v1/crops"
VALID = "scripts/distill/runs/deep_20261004/scale_gt/clips_valid_v1.json"
SPLIT = "scripts/distill/splits/fulldata_v1.json"
PRE = "weights/stable-video-diffusion-img2vid-xt-1-1/"
OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
dt = torch.bfloat16
T0 = time.time()

valid = json.load(open(VALID))["summary"]["train_valid"]
sp = json.load(open(SPLIT))
banned = set(sp["test"]) | set(sp["dev"])
clips = sys.argv[2].split(",") if len(sys.argv) > 2 else sorted(valid)
for c in clips:
    assert c in valid, f"{c} is not in the GT-valid train list"
    assert c not in banned, f"{c} is a test/dev clip"
    assert int(c) < 310, c
    assert "train_leftGT_broken" not in os.path.realpath(f"video_data/train/{c}_train.mp4"), c

vae = AutoencoderKLTemporalDecoder.from_pretrained(PRE, subfolder="vae", variant="fp16", torch_dtype=dt)
vae.requires_grad_(False)
vae.to(dtype=dt)
vae = vae.to("cuda").eval()
shim = SimpleNamespace(vae=vae, vae_scale_factor=8, image_processor=VaeImageProcessor(vae_scale_factor=8))
n_done = n_skip = 0
with torch.no_grad():
    for c in clips:
        cj = json.load(open(f"{CROPS}/{c}/clip.json"))
        for ws in cj["windows_stats"]:
            s = ws["start"]
            op = f"{OUT}/{c}_w{s:03d}.pt"
            if os.path.exists(op):
                n_skip += 1
                continue
            tgt = np.load(f"{CROPS}/{c}/w{s:03d}/tgt.npy")
            dig = hashlib.md5(np.ascontiguousarray(tgt).tobytes()).hexdigest()
            assert dig == ws["md5"]["tgt"], (c, s)
            assert tgt.shape == (14, 576, 1024, 3), tgt.shape
            x = torch.from_numpy(tgt).permute(0, 3, 1, 2).float() / 255.0
            xp = shim.image_processor.preprocess(x, height=576, width=1024)
            z = II._Pipe._encode_vae_frames(shim, xp, torch.device("cuda"), 1, False, n_frames_per_time=5)[0]
            assert z.dtype == dt and tuple(z.shape) == (14, 4, 72, 128), (z.dtype, z.shape)
            torch.save(dict(mode=z.cpu().clone(), meta=dict(clip=c, start=s, tgt_md5=dig, reg=[cj["ddy"], cj["ddx"]])), op)
            n_done += 1
            if n_done % 50 == 1:
                print(f"[enc {time.time() - T0:6.0f}s] {c} w{s:03d} done {n_done} skipped {n_skip} "
                      f"z mean {z.float().mean():+.4f} std {z.float().std():.4f}", flush=True)
print(f"ENCODE_GT_DONE done={n_done} skipped={n_skip} clips={len(clips)} {time.time() - T0:.0f}s", flush=True)
