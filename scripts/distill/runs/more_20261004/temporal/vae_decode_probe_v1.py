#!/usr/bin/env python
"""more_20261004 / temporal lane: feasibility probe for decode_chunk_size (descriptive, not a quality number).

Loads the pipeline's three GPU-resident models exactly as inpainting_inference.main does (CLIP image encoder, SVD
temporal VAE, StereoCrafter UNet; bf16, variant fp16 where main uses it), moves them to the GPU, then decodes one
14-frame window of random latents with MambaStableVideoDiffusionInpaintingPipeline.decode_latents' chunk loop
(vae.decode(latents[i:i+dcs], num_frames=len) on 1/scaling_factor latents) for each (H, W) and decode_chunk_size.
Reports torch.cuda.max_memory_allocated during the decode (weights included) and the CUDA-synchronised decode time
(median of 3 after one warm-up).  Random latents: memory and time do not depend on latent values.
usage: CUDA_VISIBLE_DEVICES=1 python vae_decode_probe_v1.py OUT.json
"""
import json
import statistics
import sys
import time

import torch
from diffusers.models.autoencoders.autoencoder_kl_temporal_decoder import AutoencoderKLTemporalDecoder
from diffusers.models.unets.unet_spatio_temporal_condition import UNetSpatioTemporalConditionModel
from transformers import CLIPVisionModelWithProjection

PRE = "/home/kawa/master_project/StereoCrafter/weights/stable-video-diffusion-img2vid-xt-1-1/"
UNET = "/home/kawa/master_project/StereoCrafter/weights/StereoCrafter/"
dt = torch.bfloat16
enc = CLIPVisionModelWithProjection.from_pretrained(PRE, subfolder="image_encoder", variant="fp16", torch_dtype=dt).cuda()
vae = AutoencoderKLTemporalDecoder.from_pretrained(PRE, subfolder="vae", variant="fp16", torch_dtype=dt).cuda()
unet = UNetSpatioTemporalConditionModel.from_pretrained(UNET, subfolder="unet_diffusers", low_cpu_mem_usage=True,
                                                        torch_dtype=dt).cuda()
for mdl in (enc, vae, unet):
    mdl.requires_grad_(False)
torch.cuda.synchronize()
w_bytes = torch.cuda.memory_allocated()
total = torch.cuda.get_device_properties(0).total_memory
print(f"[probe] weights resident {w_bytes / 2**30:.2f} GiB of {total / 2**30:.2f} GiB", flush=True)
res = dict(weights_gib=w_bytes / 2**30, total_gib=total / 2**30, rows=[])
sf = vae.config.scaling_factor
for (H, W) in ((576, 1024), (1024, 1792), (1024, 1920)):
    for dcs in (2, 7, 14):
        lat = torch.randn(14, 4, H // 8, W // 8, device="cuda", dtype=dt) / sf
        row = dict(H=H, W=W, dcs=dcs)
        try:
            times = []
            for rep in range(4):
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.no_grad():
                    outs = []
                    for i in range(0, lat.shape[0], dcs):
                        chunk = lat[i:i + dcs]
                        outs.append(vae.decode(chunk, num_frames=chunk.shape[0]).sample)
                    fr = torch.cat(outs).float()
                torch.cuda.synchronize()
                if rep:
                    times.append(time.perf_counter() - t0)
                del outs, fr
            row.update(ok=True, peak_gib=torch.cuda.max_memory_allocated() / 2**30, decode_s=statistics.median(times))
        except torch.cuda.OutOfMemoryError as e:
            row.update(ok=False, err=str(e).split("\n")[0][:160])
            torch.cuda.empty_cache()
        del lat
        res["rows"].append(row)
        print(f"[probe] {row}", flush=True)
json.dump(res, open(sys.argv[1], "w"), indent=1)
print("[probe] done", flush=True)
