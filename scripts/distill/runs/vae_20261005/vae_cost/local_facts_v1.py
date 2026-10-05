"""vae_cost lane (vae_20261005) -- CPU-only local facts used by the cost estimate.

No GPU (run with CUDA_VISIBLE_DEVICES="").  Reads weight files only; writes nothing but stdout
(captured to local_facts_v1.log by the caller).

1. Parameter counts from safetensors headers (no tensor load): StereoCrafter UNet, SVD VAE,
   Wan2.1-Fun-1.3B-InP DiT, CogVideoX-5b-I2V transformer + VAE, sd-vae-ft-mse.
2. Interface layers that a new latent space would replace (UNet conv_in / conv_out shapes) and
   their size for a 16-channel latent (computed, 3x3 kernels as in the existing layers).
3. Is SVD's VAE encoder the SD kl-f8 encoder?  Bitwise compare of all encoder + quant_conv tensors
   against stabilityai/sd-vae-ft-mse (decoder-only fine-tune, so its encoder is the original SD one).
4. Wan2.1 VAE parameter count (torch.load on CPU, weights_only).
"""
import glob
import json
import os
import struct

import torch
from safetensors.torch import load_file

SC = "/home/kawa/master_project/StereoCrafter"
TP = "/home/kawa/master_project/third_party"
FILES = {
    "stereocrafter_unet": [f"{SC}/weights/StereoCrafter/unet_diffusers/diffusion_pytorch_model.safetensors"],
    "svd_vae": [f"{SC}/weights/stable-video-diffusion-img2vid-xt-1-1/vae/diffusion_pytorch_model.safetensors"],
    "wan21_fun_1p3b_inp_dit": [f"{TP}/ROSE/models/Wan2.1-Fun-1.3B-InP/diffusion_pytorch_model.safetensors"],
    "cogvideox_5b_i2v_transformer": sorted(glob.glob(f"{TP}/VideoPainter/ckpt/CogVideoX-5b-I2V/transformer/*.safetensors")),
    "cogvideox_5b_i2v_vae": [f"{TP}/VideoPainter/ckpt/CogVideoX-5b-I2V/vae/diffusion_pytorch_model.safetensors"],
    "sd_vae_ft_mse": [f"{TP}/DiffuEraser/weights/sd-vae-ft-mse/diffusion_pytorch_model.safetensors"],
}


def header(fp):
    with open(fp, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def numel(shape):
    n = 1
    for s in shape:
        n *= s
    return n


out = {"param_counts_M": {}, "groups_M": {}, "shapes": {}}
for name, files in FILES.items():
    tot, groups = 0, {}
    for fp in files:
        h = header(fp)
        for k, v in h.items():
            if k == "__metadata__":
                continue
            n = numel(v["shape"])
            tot += n
            g = k.split(".")[0]
            groups[g] = groups.get(g, 0) + n
            if name == "stereocrafter_unet" and k in ("conv_in.weight", "conv_out.weight"):
                out["shapes"][f"{name}.{k}"] = v["shape"]
            if name == "wan21_fun_1p3b_inp_dit" and k in ("patch_embedding.weight", "head.head.weight"):
                out["shapes"][f"{name}.{k}"] = v["shape"]
            if name == "cogvideox_5b_i2v_transformer" and k == "patch_embed.proj.weight":
                out["shapes"][f"{name}.{k}"] = v["shape"]
    out["param_counts_M"][name] = round(tot / 1e6, 2)
    out["groups_M"][name] = {k: round(v / 1e6, 2) for k, v in sorted(groups.items(), key=lambda x: -x[1])[:6]}

# temporal vs spatial share of the StereoCrafter UNet (StereoCrafter's paper fine-tuned only spatial layers)
h = header(FILES["stereocrafter_unet"][0])
temporal = sum(numel(v["shape"]) for k, v in h.items() if k != "__metadata__" and ("temporal" in k or "time_mixer" in k))
total = sum(numel(v["shape"]) for k, v in h.items() if k != "__metadata__")
out["stereocrafter_unet_temporal_named_params_M"] = round(temporal / 1e6, 2)
out["stereocrafter_unet_nontemporal_params_M"] = round((total - temporal) / 1e6, 2)

# interface layers for a 16-channel latent (computed): conv_in 3x3 over (16 noisy + 16 cond + 1 mask), conv_out 3x3 -> 16
ci = out["shapes"]["stereocrafter_unet.conv_in.weight"]  # [320, 9, 3, 3]
co = out["shapes"]["stereocrafter_unet.conv_out.weight"]  # [4, 320, 3, 3]
out["interface_now_params"] = numel(ci) + ci[0] + numel(co) + co[0]
out["interface_16ch_params"] = (320 * 33 * 9 + 320) + (16 * 320 * 9 + 16)
out["interface_16ch_share_of_unet"] = out["interface_16ch_params"] / total

# 3. SVD VAE encoder == SD kl-f8 encoder ?
a = load_file(FILES["svd_vae"][0])
b = load_file(FILES["sd_vae_ft_mse"][0])
rename = {".to_q.": ".query.", ".to_k.": ".key.", ".to_v.": ".value.", ".to_out.0.": ".proj_attn."}
n_cmp, n_exact, maxd, missing = 0, 0, 0.0, []
for k in sorted(a):
    if not (k.startswith("encoder.") or k.startswith("quant_conv")):
        continue
    kb = k
    if kb not in b:
        for old, new in rename.items():
            kb = kb.replace(old, new)
    if kb not in b:
        missing.append(k)
        continue
    x, y = a[k].float().reshape(-1), b[kb].float().reshape(-1)
    n_cmp += 1
    d = (x - y).abs().max().item() if x.numel() == y.numel() else float("inf")
    n_exact += int(d == 0.0)
    maxd = max(maxd, d)
out["svd_vs_sdvaeftmse_encoder"] = {"compared": n_cmp, "exact_equal": n_exact, "max_abs_diff": maxd, "missing": missing}
out["svd_decoder_tensors"] = sum(1 for k in a if k.startswith("decoder."))
out["sd_decoder_tensors"] = sum(1 for k in b if k.startswith("decoder."))

# 4. Wan2.1 VAE
wan_vae = f"{TP}/ROSE/models/Wan2.1-Fun-1.3B-InP/Wan2.1_VAE.pth"
sd = torch.load(wan_vae, map_location="cpu", weights_only=True)
if isinstance(sd, dict) and "state_dict" in sd:
    sd = sd["state_dict"]
out["param_counts_M"]["wan21_vae"] = round(sum(v.numel() for v in sd.values() if hasattr(v, "numel")) / 1e6, 2)
conv1 = [(k, list(v.shape)) for k, v in sd.items() if k in ("conv1.weight", "conv2.weight", "encoder.head.2.weight", "decoder.conv1.weight")]
out["shapes"]["wan21_vae_interface"] = conv1

print(json.dumps(out, indent=1))
