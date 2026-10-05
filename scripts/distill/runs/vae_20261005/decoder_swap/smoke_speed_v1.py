#!/usr/bin/env python
"""vae_20261005 / decoder_swap -- feasibility smoke (NO quality number): speed, peak memory and determinism of each candidate
decoder on ONE captured latent window (0040_deliv_cap w000).  Prints timings, max|out1-out2| for the consistency decoder with
the same seed, and output ranges only.
usage: CUDA_VISIBLE_DEVICES=0 flock /tmp/claude-gpu0.lock python smoke_speed_v1.py
"""
import time, torch
from diffusers import AutoencoderKL, ConsistencyDecoderVAE
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
HUB = "/mnt/ssd_data/vae_20261005/decoder_swap/hf_home/hub"
CD = f"{HUB}/models--openai--consistency-decoder/snapshots/63b7a48896d92b6f56772f4111d0860b1bee3dd3"
FTMSE = "/home/kawa/master_project/third_party/DiffuEraser/weights/sd-vae-ft-mse"
lat = torch.load("/mnt/ssd_data/deep_20261004/decoder_ft/latents/0040_deliv_cap/w000.pt", map_location="cpu")[0]  # [14,4,72,128] bf16 scaled
print("latent", lat.shape, lat.dtype, float(lat.float().std()))
sf = 0.18215
def sync(): torch.cuda.synchronize()
cd = ConsistencyDecoderVAE.from_pretrained(CD, torch_dtype=torch.float16).to("cuda").eval()
print("cd scaling_factor", cd.config.scaling_factor, "means", cd.means.flatten().tolist(), "stds", cd.stds.flatten().tolist())
z = (lat[:2].float() / sf).half().cuda()
outs = []
with torch.no_grad():
    for rep in range(3):
        torch.cuda.reset_peak_memory_stats(); sync(); t0 = time.time()
        o = []
        for j in range(2):
            g = torch.Generator(device="cuda").manual_seed(20261005)
            o.append(cd.decode(z[j:j + 1], generator=g, num_inference_steps=2).sample)
        sync(); dt = time.time() - t0
        o = torch.cat(o).float()
        outs.append(o)
        print(f"cd rep{rep}: {dt / 2:.3f} s/frame, peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB, range [{float(o.min()):.3f},{float(o.max()):.3f}]")
print("cd same-seed max|d| rep0-rep1", float((outs[0] - outs[1]).abs().max()), "rep0-rep2", float((outs[0] - outs[2]).abs().max()))
with torch.no_grad():
    g = torch.Generator(device="cuda").manual_seed(7)
    o2 = cd.decode(z[0:1], generator=g, num_inference_steps=2).sample.float()
print("cd other-seed max|d| frame0", float((outs[0][0:1] - o2).abs().max()), "mean|d|", float((outs[0][0:1] - o2).abs().mean()))
del cd; torch.cuda.empty_cache()
kl = AutoencoderKL.from_pretrained(FTMSE, torch_dtype=torch.float32).to("cuda").eval()
print("ftmse scaling_factor", kl.config.scaling_factor)
with torch.no_grad():
    for rep in range(2):
        torch.cuda.reset_peak_memory_stats(); sync(); t0 = time.time()
        o = kl.decode(1 / kl.config.scaling_factor * lat[:2].float().cuda()).sample
        sync(); dt = time.time() - t0
        print(f"ftmse fp32 rep{rep}: {dt / 2:.3f} s/frame (batch 2), peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
print("SMOKE_DONE")
