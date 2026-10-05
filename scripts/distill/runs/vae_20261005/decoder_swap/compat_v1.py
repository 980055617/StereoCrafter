#!/usr/bin/env python
"""vae_20261005 / decoder_swap -- STEP 1 COMPATIBILITY: is SVD's latent space the SD kl-f8 latent space?

(A) WEIGHTS (CPU): every encoder.* and quant_conv.* tensor of the SVD VAE (fp32 file and the deployed fp16 variant) vs
    the encoders shipped with sd-vae-ft-mse (fp32), sd-vae-ft-ema (fp32), the SD1.5 base VAE (VideoPainter copy, fp16)
    and OpenAI's consistency decoder (fp16).  Reported per pair: number of tensors, number bit-identical (after casting
    the higher-precision side to the lower precision, round-to-nearest), max |diff|, max relative L2 diff.
    Decoder weights of the SD-family decoders are compared too (to confirm they are DIFFERENT decoders).
(B) LATENTS (GPU, run under the GPU lock): 14 real-right-eye frames of two dev clips (0040 AVP, 0268 iPhone; gt_dev
    window, frames 0..13) encoded by (i) the deployed SVD encode (VAE fp16 variant -> bf16, image_processor.preprocess,
    _encode_vae_frames n_frames_per_time=5, mode) twice (run-to-run floor), (ii) the same path with the sd-vae-ft-mse
    AutoencoderKL (loaded bf16; mode), (iii) the same with the SVD VAE in fp32 and ft-mse in fp32.  Reported: max |diff|
    and relative RMS diff of the unscaled latents.
usage: CUDA_VISIBLE_DEVICES=0 flock /tmp/claude-gpu0.lock python compat_v1.py <out_json>
"""
import json
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
from safetensors import safe_open

REPO = "/home/kawa/master_project/StereoCrafter"
sys.path.insert(0, REPO)
os.chdir(REPO)
OUTJ = sys.argv[1]
assert not os.path.exists(OUTJ), f"refusing to overwrite {OUTJ}"
HUB = "/mnt/ssd_data/vae_20261005/decoder_swap/hf_home/hub"
P = {
    "svd_fp32": "/mnt/ssd_data/stereocrafter_weights/stable-video-diffusion-img2vid-xt-1-1/vae/diffusion_pytorch_model.safetensors",
    "svd_fp16": "/mnt/ssd_data/stereocrafter_weights/stable-video-diffusion-img2vid-xt-1-1/vae/diffusion_pytorch_model.fp16.safetensors",
    "ftmse": "/home/kawa/master_project/third_party/DiffuEraser/weights/sd-vae-ft-mse/diffusion_pytorch_model.safetensors",
    "ftema": f"{HUB}/models--stabilityai--sd-vae-ft-ema/snapshots/f04b2c4b98319346dad8c65879f680b1997b204a/diffusion_pytorch_model.safetensors",
    "sd15": "/home/kawa/master_project/third_party/VideoPainter/ckpt/sd15_base/vae/diffusion_pytorch_model.fp16.safetensors",
    "cd": f"{HUB}/models--openai--consistency-decoder/snapshots/63b7a48896d92b6f56772f4111d0860b1bee3dd3/diffusion_pytorch_model.safetensors",
}
T0 = time.time()


def log(*a):
    print(f"[compat {time.time() - T0:6.1f}s]", *a, flush=True)


def load(name, prefixes):
    with safe_open(P[name], "pt") as f:
        return {k: f.get_tensor(k) for k in f.keys() if k.split(".")[0] in prefixes}


def compare(a, b):
    ks = sorted(set(a) & set(b))
    only = sorted(set(a) ^ set(b))
    n_id, mx, mrel, worst = 0, 0.0, 0.0, None
    for k in ks:
        x, y = a[k], b[k]
        assert x.shape == y.shape, (k, x.shape, y.shape)
        lo = x.dtype if torch.finfo(x.dtype).bits <= torch.finfo(y.dtype).bits else y.dtype
        if torch.equal(x.to(lo), y.to(lo)):
            n_id += 1
        d = (x.double() - y.double())
        m = float(d.abs().max())
        r = float(d.norm() / max(x.double().norm(), 1e-30))
        if m > mx:
            mx, worst = m, k
        mrel = max(mrel, r)
    return dict(n_common=len(ks), n_only_one_side=len(only), only_examples=only[:4], n_bit_identical_at_lower_precision=n_id,
                max_abs_diff=mx, max_rel_l2_diff=mrel, worst_tensor=worst)


res = dict(paths=P, weights={}, latents={})
ENC = ("encoder", "quant_conv")
enc = {n: load(n, ENC) for n in P}
for n in P:
    log(f"{n}: {len(enc[n])} encoder/quant_conv tensors, dtype {next(iter(enc[n].values())).dtype}")
for n in ["svd_fp16", "ftmse", "ftema", "sd15", "cd"]:
    res["weights"][f"encoder svd_fp32 vs {n}"] = c = compare(enc["svd_fp32"], enc[n])
    log(f"ENCODER svd_fp32 vs {n:8s}: {c['n_bit_identical_at_lower_precision']}/{c['n_common']} bit-identical at lower precision, "
        f"max|d| {c['max_abs_diff']:.3e}, max rel L2 {c['max_rel_l2_diff']:.3e} (worst {c['worst_tensor']}); "
        f"{c['n_only_one_side']} keys on one side only {c['only_examples']}")
res["weights"]["encoder ftmse vs ftema"] = c = compare(enc["ftmse"], enc["ftema"])
log(f"ENCODER ftmse vs ftema: {c['n_bit_identical_at_lower_precision']}/{c['n_common']} identical, max|d| {c['max_abs_diff']:.3e}")
DEC = ("decoder", "post_quant_conv")
dec = {n: load(n, DEC) for n in ["ftmse", "ftema", "sd15"]}
for a, b in [("ftmse", "ftema"), ("ftmse", "sd15"), ("ftema", "sd15")]:
    res["weights"][f"decoder {a} vs {b}"] = c = compare(dec[a], dec[b])
    log(f"DECODER {a} vs {b}: {c['n_bit_identical_at_lower_precision']}/{c['n_common']} bit-identical, max|d| {c['max_abs_diff']:.3e}, "
        f"max rel L2 {c['max_rel_l2_diff']:.3e}")

# ------------------------------------------------------------------------------------------- (B) latents
import inpainting_inference as II  # noqa: E402
from diffusers import AutoencoderKL  # noqa: E402
from diffusers.image_processor import VaeImageProcessor  # noqa: E402
from diffusers.models.autoencoders.autoencoder_kl_temporal_decoder import AutoencoderKLTemporalDecoder  # noqa: E402

PRE = "weights/stable-video-diffusion-img2vid-xt-1-1/"
torch.backends.cudnn.benchmark = False


def svd_vae(dtype, variant):
    v = AutoencoderKLTemporalDecoder.from_pretrained(PRE, subfolder="vae", variant=variant, torch_dtype=dtype)
    v.requires_grad_(False)
    return v.to(dtype=dtype).to("cuda").eval()


def kl_vae(path, dtype):
    v = AutoencoderKL.from_pretrained(path, torch_dtype=dtype)
    v.requires_grad_(False)
    return v.to(dtype=dtype).to("cuda").eval()


def encode(vae, x):
    shim = SimpleNamespace(vae=vae, vae_scale_factor=8, image_processor=VaeImageProcessor(vae_scale_factor=8))
    xp = shim.image_processor.preprocess(x, height=x.shape[2], width=x.shape[3])
    with torch.no_grad():
        z = II._Pipe._encode_vae_frames(shim, xp, torch.device("cuda"), 1, False, n_frames_per_time=5)
    return z.float().cpu()


def diff(a, b):
    d = (a.double() - b.double())
    return dict(max_abs=float(d.abs().max()), rel_rms=float(d.pow(2).mean().sqrt() / a.double().pow(2).mean().sqrt()),
                mean_a=float(a.mean()), std_a=float(a.std()), mean_b=float(b.mean()), std_b=float(b.std()))


FTMSE_DIR = os.path.dirname(P["ftmse"])
for clip in ["0040", "0268"]:
    X = np.load(f"/mnt/ssd_data/deep_20261004/decoder_ft/gt_dev/{clip}_TR.npy", mmap_mode="r")
    x = torch.from_numpy(np.ascontiguousarray(X[0:14])).permute(0, 3, 1, 2).float() / 255.0
    out = {}
    v = svd_vae(torch.bfloat16, "fp16")
    z_svd = encode(v, x)
    z_svd2 = encode(v, x)
    del v
    v = kl_vae(FTMSE_DIR, torch.bfloat16)
    z_mse = encode(v, x)
    del v
    v = svd_vae(torch.float32, None)
    z_svd32 = encode(v, x)
    del v
    v = kl_vae(FTMSE_DIR, torch.float32)
    z_mse32 = encode(v, x)
    del v
    torch.cuda.empty_cache()
    out["svd_bf16 run1 vs run2 (floor)"] = diff(z_svd, z_svd2)
    out["svd_bf16 (deployed) vs ftmse_bf16"] = diff(z_svd, z_mse)
    out["svd_fp32 vs ftmse_fp32"] = diff(z_svd32, z_mse32)
    out["svd_bf16 (deployed) vs svd_fp32 (precision)"] = diff(z_svd32, z_svd)
    res["latents"][clip] = out
    for k, d in out.items():
        log(f"LATENT {clip} {k:42s} max|d| {d['max_abs']:.4e} relRMS {d['rel_rms']:.4e}  (mean/std a {d['mean_a']:+.4f}/{d['std_a']:.4f})")
res["seconds"] = time.time() - T0
json.dump(res, open(OUTJ, "w"), indent=1)
log(f"wrote {OUTJ}")
print("COMPAT_DONE", flush=True)
