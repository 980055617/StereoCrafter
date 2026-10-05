#!/usr/bin/env python
"""deep_20261004 / decoder_ft lane -- GATE T1 on the decoder's training data (scaling / frame-correspondence check).

For K windows (fixed seed) of encode_gt_v1.py's latents:
  (a) stock bf16 decoder via the deployed decode_latents on bf16(mode*sf)  ->  vs tgt.npy frames: PSNR / LPIPS-Alex
      (must look like blur_diag's VAE_GT round trip: PSNR ~ 25-35 dB, LPIPS ~ 0.03-0.15; a scaling or frame-order error
      gives PSNR < 15 dB)
  (b) a FRESH deployed encode of the same tgt frames, decoded the same way -> max |a-b| in uint8 levels (encoder
      non-determinism floor only; must be <= 3 levels at the 99.9th percentile)
  (c) the TRAINER's own input path (z * SF -> 1/SF * z_s in bf16) is bit-equal to decode_latents' path on the same tensor
usage: CUDA_VISIBLE_DEVICES=1 flock /tmp/claude-gpu1.lock python gate_traindata_v1.py <out_txt>
"""
import os
import random
import sys
from types import SimpleNamespace

import numpy as np
import torch

REPO = "/home/kawa/master_project/StereoCrafter"
sys.path.insert(0, REPO)
os.chdir(REPO)
import inpainting_inference as II  # noqa: E402
import lpips  # noqa: E402
from diffusers.image_processor import VaeImageProcessor  # noqa: E402
from diffusers.models.autoencoders.autoencoder_kl_temporal_decoder import AutoencoderKLTemporalDecoder  # noqa: E402

OUT = sys.argv[1]
assert not os.path.exists(OUT)
LATD = "/mnt/ssd_data/deep_20261004/decoder_ft/gt_latents"
CROPS = "/mnt/ssd_data/deep_20261004/scale_gt/cache_v1/crops"
PRE = "weights/stable-video-diffusion-img2vid-xt-1-1/"
dt = torch.bfloat16
SF = 0.18215
vae = AutoencoderKLTemporalDecoder.from_pretrained(PRE, subfolder="vae", variant="fp16", torch_dtype=dt)
vae.requires_grad_(False)
vae.to(dtype=dt)
vae = vae.to("cuda").eval()
shim = SimpleNamespace(vae=vae, vae_scale_factor=8, image_processor=VaeImageProcessor(vae_scale_factor=8))
net = lpips.LPIPS(net="alex", verbose=False).cuda().eval()
files = sorted(f for f in os.listdir(LATD) if f.endswith(".pt"))
random.seed(7)
pick = random.sample(files, 4)
lines = [f"GATE T1 training data, {len(files)} latent windows; checked {pick}"]
ok = True


def dec(lat):
    vf = II._Pipe.decode_latents(shim, lat, num_frames=lat.shape[1], decode_chunk_size=2)
    vf = II.tensor2vid(vf, shim.image_processor, output_type="pil")[0]
    return np.stack([np.array(im) for im in vf])


with torch.no_grad():
    for f in pick:
        d = torch.load(f"{LATD}/{f}", map_location="cpu")
        c, s = d["meta"]["clip"], d["meta"]["start"]
        tgt = np.load(f"{CROPS}/{c}/w{s:03d}/tgt.npy")
        z = d["mode"].to("cuda")
        lat = (z.float() * SF).to(dt).unsqueeze(0)                     # blur_diag vae_path / pipeline x0 space
        z_s = z * SF                                                     # trainer path
        same = torch.equal(z_s, lat[0])
        za = dec(lat)
        ps = [10 * np.log10(255.0 ** 2 / max(((za[j].astype(np.float64) - tgt[j]) ** 2).mean(), 1e-9)) for j in range(14)]
        A = torch.from_numpy(za).permute(0, 3, 1, 2).float().cuda() / 255.
        T = torch.from_numpy(tgt).permute(0, 3, 1, 2).float().cuda() / 255.
        lp = float(net(A * 2 - 1, T * 2 - 1).mean())
        x = torch.from_numpy(tgt).permute(0, 3, 1, 2).float() / 255.0
        xp = shim.image_processor.preprocess(x, height=576, width=1024)
        z2 = II._Pipe._encode_vae_frames(shim, xp, torch.device("cuda"), 1, False, n_frames_per_time=5)
        zb = dec((z2.float() * SF).to(dt))
        dif = np.abs(za.astype(np.int16) - zb.astype(np.int16))
        p999 = float(np.percentile(dif, 99.9))
        good = (min(ps) > 20.0) and (lp < 0.2) and (p999 <= 3) and same
        ok &= good
        lines.append(f"{c} w{s:03d}: decode-vs-tgt PSNR mean {np.mean(ps):.2f} min {min(ps):.2f} dB, LPIPS-alex {lp:.4f}; "
                     f"fresh-encode diff p99.9 {p999:.1f} max {int(dif.max())} levels; trainer scaling path bit-equal {same} "
                     f"-> {'OK' if good else 'FAIL'}")
lines.append(f"GATE T1 {'PASS' if ok else 'FAIL'}")
open(OUT, "w").write("\n".join(lines) + "\n")
print("\n".join(lines), flush=True)
