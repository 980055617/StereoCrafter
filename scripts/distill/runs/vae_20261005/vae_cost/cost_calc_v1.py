"""vae_cost lane (vae_20261005) -- cost arithmetic for "replace the whole VAE", CPU only.

Every input carries its source (URL, or a file in this repo).  Every derived number is an ESTIMATE whose formula
is written next to it.  Output: COST_TABLE_v1.json (this dir).  No GPU, no network; the literature numbers were read
on 2026-10-05 with WebFetch (URLs below) and the local numbers come from the files named.

Conventions
  * "4090-h" = hours of ONE RTX 4090.  Calendar days on this machine = 4090-h / 2 GPUs / 24, i.e. perfect 2-GPU
    scaling and exclusive use of both cards (both are shared with other lanes) -> every calendar figure is a LOWER bound.
  * Conversion of published GPU-hours to 4090-h uses NVIDIA spec ratios (dense BF16 tensor throughput; memory
    bandwidth as the second bound).  A spec ratio is optimistic for a 24 GB card that needs offload/checkpointing.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

SRC = {
    "ada_wp": "https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf",
    "a100": "https://www.nvidia.com/en-us/data-center/a100/",
    "h100": "https://www.nvidia.com/en-us/data-center/h100/",
    "a800": "https://lenovopress.lenovo.com/lp1813-thinksystem-nvidia-a800-pcie-gpu",
    "v100": "https://images.nvidia.com/content/technologies/volta/pdf/tesla-volta-v100-datasheet-letter-fnl-web.pdf",
    "zero": "https://arxiv.org/abs/1910.02054",
    "kaplan": "https://arxiv.org/abs/2001.08361",
    "ckpt": "https://arxiv.org/abs/1604.06174",
    "stereocrafter": "https://arxiv.org/pdf/2409.07447",
    "m2svid": "https://arxiv.org/html/2505.16565v2",
    "stereoworld": "https://arxiv.org/html/2512.09363v2",
    "stereopilot": "https://arxiv.org/html/2512.16915v1",
    "dcvideogen": "https://arxiv.org/html/2509.25182v1",
    "dcgen": "https://hanlab.mit.edu/projects/dc-gen",
    "pixartsigma": "https://arxiv.org/html/2403.04692v2",
    "sd3": "https://arxiv.org/html/2403.03206",
    "hunyuan": "https://arxiv.org/html/2412.03603",
    "wan21": "https://github.com/Wan-Video/Wan2.1",
    "wan21_generate_py": "https://raw.githubusercontent.com/Wan-Video/Wan2.1/main/generate.py",
    "videox_fun": "https://github.com/aigc-apps/VideoX-Fun",
    "cogft": "https://github.com/THUDM/CogVideo/tree/main/finetune",
    "musubi": "https://github.com/kohya-ss/musubi-tuner",
    "rose": "https://arxiv.org/pdf/2508.18633",
    "videopainter": "https://arxiv.org/html/2503.05639",
    "spatialdreamer": "https://arxiv.org/html/2411.11934v2",
    "grt": "https://huggingface.co/papers/2607.05354.md",
    "stereo4d": "https://arxiv.org/html/2412.09621v2",
    "svd_card": "https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt",
    "sdvae_ftmse": "https://huggingface.co/stabilityai/sd-vae-ft-mse",
    "civitai_translator": "https://civitai.com/articles/18185",
    "ostris16ch": "https://huggingface.co/ostris/vae-kl-f8-d16",
    "pixartalpha_vae_cfg": "https://huggingface.co/PixArt-alpha/PixArt-XL-2-1024-MS/raw/main/vae/config.json",
    "sdxl_vae_cfg_local": "/home/kawa/master_project/third_party/VideoPainter/ckpt/sdxl_inpainting/vae/config.json",
    # local
    "timing": "scripts/distill/runs/finalcheck_20261004/speed/TABLE_TIMING_final.txt",
    "decoder_ft": "scripts/distill/runs/deep_20261004/decoder_ft/RESULTS.txt",
    "scale_gt": "scripts/distill/runs/deep_20261004/scale_gt/RESULTS.txt",
    "clips_valid": "scripts/distill/runs/deep_20261004/scale_gt/clips_valid_v1.json",
    "blur_diag": "scripts/distill/runs/deep_20261004/blur_diag/TABLE_BLUR_DIAG_r2.txt",
    "local_facts": "scripts/distill/runs/vae_20261005/vae_cost/local_facts_v1.log",
    "changelog": "docs/agents/model-change-log.md (2026-09-29 entry, positive controls)",
}

# ---------------------------------------------------------------- inputs: GPU specs (dense numbers)
TFLOPS = {"4090": 165.2, "A100": 312.0, "A800": 312.0, "H100_SXM": 1979.0 / 2, "V100": 125.0}
TFLOPS_SRC = {"4090": "ada_wp (Peak BF16 Tensor TFLOPS w/ FP32 accumulate 165.2/330.4)",
              "A100": "a100 (BF16 312 | 624*, *=sparsity)", "A800": "a800 (BF16 312)",
              "H100_SXM": "h100 (BF16 1,979* with sparsity -> dense = /2)", "V100": "v100 (Tensor 125)"}
BW_GBs = {"4090": 1008.0, "A100": 2039.0, "H100_SXM": 3350.0}  # ada_wp, a100 (SXM), h100 (SXM 3.35 TB/s)

ratio_compute = {g: TFLOPS[g] / TFLOPS["4090"] for g in TFLOPS}
ratio_bw = {g: BW_GBs[g] / BW_GBs["4090"] for g in BW_GBs}

# ---------------------------------------------------------------- inputs: local measurements
UNET_FWD_14F_S = 4.21 / 8          # timing: origin_g100_s8 unet/win 4.21 s, 8 UNet calls, batch 1, 14 frames, 576x1024
K_TRAIN = (3.0, 4.0)               # kaplan: train ~ 3x forward; ckpt: +1 forward for activation recompute
UNET_PARAMS_M = 1524.63            # local_facts
WAN13_PARAMS_M = 1564.43           # local_facts
COG5_PARAMS_M = 5625.09            # local_facts
BYTES_PER_PARAM_ADAM_MIXED = 16    # zero: 2 (fp16 w) + 2 (fp16 g) + 12 (fp32 master, m, v)
CLIPS_LOCAL, FRAMES_LOCAL = 291, 43843   # clips_valid (sum of train_frames of the 291 GT-valid clips)
DECODER_FT_S, DECODER_ENCODE_S = 5908.0, 890.0   # decoder_ft

out = {"sources": SRC, "assumption_labels": {}, "inputs": {}, "derived": {}}
out["inputs"]["tflops_dense"] = {g: {"value": TFLOPS[g], "src": TFLOPS_SRC[g]} for g in TFLOPS}
out["inputs"]["mem_bw_GBs"] = BW_GBs
out["derived"]["spec_ratio_vs_4090"] = {
    "compute": {g: round(v, 3) for g, v in ratio_compute.items()},
    "bandwidth": {g: round(v, 3) for g, v in ratio_bw.items()},
    "formula": "ratio = spec(GPU) / spec(RTX 4090); 1 GPU-h on GPU ~= ratio 4090-h (optimistic for a 24 GB card)",
}

# ---------------------------------------------------------------- memory: does a full fine-tune fit in 24 GB?
def full_ft_state_gb(params_m):
    return params_m * 1e6 * BYTES_PER_PARAM_ADAM_MIXED / 1e9

out["derived"]["full_ft_weights_grads_adam_GB_before_activations"] = {
    "stereocrafter_unet": round(full_ft_state_gb(UNET_PARAMS_M), 1),
    "wan21_1p3b_dit": round(full_ft_state_gb(WAN13_PARAMS_M), 1),
    "cogvideox_5b_dit": round(full_ft_state_gb(COG5_PARAMS_M), 1),
    "per_gpu_with_zero2_over_2gpus_unet": round(UNET_PARAMS_M * 1e6 * (2 + (2 + 12) / 2) / 1e9, 1),
    "formula": "params x 16 B (ZeRO paper, mixed-precision Adam); ZeRO-2 over 2 GPUs: 2 B + (2+12)/2 B per param per GPU",
    "note": "24 GB card: full fine-tune of any of the three needs sharding + CPU optimizer offload or LoRA; the 14/25-frame "
            "576x1024 whole-UNet fit is UNMEASURED here (the project's whole-UNet runs used 2-frame chunks).",
}

# ---------------------------------------------------------------- per-sample training time on a 4090 (SVD UNet)
step14 = tuple(k * UNET_FWD_14F_S for k in K_TRAIN)
step25 = tuple(s * 25 / 14 for s in step14)
out["derived"]["svd_unet_train_s_per_sample_4090"] = {
    "fwd_14f_s": round(UNET_FWD_14F_S, 3),
    "train_14f_s": [round(x, 2) for x in step14],
    "train_25f_s": [round(x, 2) for x in step25],
    "formula": "k x t_fwd(14 frames, 576x1024), k = 3 (fwd+bwd) .. 4 (+recompute); 25 frames scaled linearly",
    "label": "ESTIMATE, compute only: no optimizer/CPU-offload, data loading, eval or failed runs",
}

def days_on_2x4090(h4090):
    return h4090 / 2 / 24

# Route 1a anchor A: StereoCrafter's own fine-tune volume (8 A100 x bs1 x 26K it, 25x576x1024)
sc_samples = 26000 * 8
sc_h = tuple(sc_samples * s / 3600 for s in step25)
out["derived"]["route1a_anchor_stereocrafter_volume"] = {
    "samples": sc_samples,
    "4090_h": [round(x) for x in sc_h],
    "calendar_days_2x4090": [round(days_on_2x4090(x), 1) for x in sc_h],
    "formula": "26,000 iterations x 8 GPUs x batch 1 (stereocrafter) x train_25f_s / 3600",
    "label": "ESTIMATE, lower bound per training run. FLOOR ONLY: this is StereoCrafter's fine-tune volume INSIDE SVD's own "
             "latent space (no latent swap); not a latent-swap estimate",
}
# Route 1a anchor B: DC-VideoGen's adaptation volume for Wan-2.1-1.3B (output head <=4k x bs32 + LoRA 20k x bs32);
# the 20k x bs4 patch-embedder MSE stage only runs the embedder and is left out.
dcv_samples = 4000 * 32 + 20000 * 32
dcv_h14 = tuple(dcv_samples * s / 3600 for s in step14)
dcv_h25 = tuple(dcv_samples * s / 3600 for s in step25)
out["derived"]["route1a_anchor_dcvideogen_volume"] = {
    "samples": dcv_samples,
    "4090_h_14f": [round(x) for x in dcv_h14],
    "4090_h_25f": [round(x) for x in dcv_h25],
    "calendar_days_2x4090_14f": [round(days_on_2x4090(x), 1) for x in dcv_h14],
    "calendar_days_2x4090_25f": [round(days_on_2x4090(x), 1) for x in dcv_h25],
    "formula": "(4k head steps + 20k LoRA steps) x batch 32 (dcvideogen, Wan-2.1-1.3B) x train_s_per_sample / 3600",
    "label": "ESTIMATE, lower bound per training run; the LATENT-SWAP anchor (their samples are Wan videos, used here only "
             "as a step-count anchor). Compute-only k x forward method -- NOT comparable 1:1 with route 2's converted wall time",
}

# Route 2 anchor: StereoWorld (Wan2.1-T2V-1.3B, LoRA r128, 8 A800, ~11 days, 142,520 clips, 480x832x81)
sw_gpu_h = 8 * 11 * 24
sw_4090_h = (sw_gpu_h * ratio_compute["A800"], sw_gpu_h * ratio_bw["A100"])
out["derived"]["route2_anchor_stereoworld"] = {
    "A800_gpu_h": sw_gpu_h,
    "A800_gpu_s_per_clip": round(sw_gpu_h * 3600 / 142520, 1),
    "4090_h": [round(x) for x in sw_4090_h],
    "calendar_days_2x4090": [round(days_on_2x4090(x), 1) for x in sw_4090_h],
    "formula": "8 GPUs x 11 days x 24 h; x A800->4090 compute ratio (and A100 bandwidth ratio as 2nd bound); /2 GPUs /24",
    "label": "ESTIMATE (spec-ratio conversion of a published wall time); same data volume assumed",
}
# Same k x forward method applied to StereoWorld's volume, to expose the method gap (naive compute vs reported wall time).
# Wan README: T2V-1.3B makes a 5-s 480P video on an RTX 4090 in ~4 min with --offload_model True --t5_cpu;
# generate.py defaults: sample_steps 50 (t2v), guide scale 5.0 (README example 6) -> CFG = 2 DiT forwards per step.
wan_fwd_s_upper = 4 * 60 / (50 * 2)   # <= 2.4 s per 81-frame 832x480 forward (upper bound: includes offload, T5 on CPU, decode)
sw_naive_h = tuple(142520 * k * wan_fwd_s_upper / 3600 for k in K_TRAIN)
out["derived"]["route2_stereoworld_naive_vs_reported"] = {
    "wan13_fwd_s_upper_4090": wan_fwd_s_upper,
    "naive_4090_h": [round(x) for x in sw_naive_h],
    "naive_calendar_days_2x4090": [round(days_on_2x4090(x), 1) for x in sw_naive_h],
    "reported_wall_converted_4090_h": [round(x) for x in sw_4090_h],
    "reported_over_naive_ratio": [round(min(sw_4090_h) / max(sw_naive_h), 1), round(max(sw_4090_h) / min(sw_naive_h), 1)],
    "formula": "142,520 clips x k (3..4) x 2.4 s / 3600; ratio = converted reported wall time / naive estimate",
    "label": "ESTIMATE. Shows that a k x forward compute floor can sit ~an order of magnitude below a real pipeline's "
             "wall time (StereoWorld also conditions on the left view, adds a geometry loss, data loading, etc.)",
}

# Other published adaptation costs converted the same way (for scale only)
conv = {}
for name, gpu, gpu_h in [("dcvideogen_wan14b", "H100_SXM", 10 * 24), ("dcgen_flux12b", "H100_SXM", 40 * 24),
                         ("pixartsigma_vae_swap_4ch_to_4ch", "V100", 5 * 24)]:
    lo = gpu_h * min(ratio_compute[gpu], ratio_bw.get(gpu, ratio_compute[gpu]))
    hi = gpu_h * max(ratio_compute[gpu], ratio_bw.get(gpu, ratio_compute[gpu]))
    conv[name] = {"gpu": gpu, "gpu_h": gpu_h, "4090_h": [round(lo), round(hi)],
                  "calendar_days_2x4090": [round(days_on_2x4090(lo), 1), round(days_on_2x4090(hi), 1)]}
conv["note"] = ("dcvideogen/dcgen adapt 14B/12B models that do not train on 24 GB cards at all; pixartsigma swapped "
                "PixArt-alpha's VAE (latent_channels 4, scaling 0.18215: pixartalpha_vae_cfg) for SDXL's VAE (latent_channels "
                "4: sdxl_vae_cfg_local) -> a same-shape swap, not evidence for a 4->16 channel change")
out["derived"]["published_costs_in_4090_terms"] = conv

# Route 3: measured here
out["derived"]["route3_measured_decoder_ft"] = {
    "train_gpu_h": round(DECODER_FT_S / 3600, 2), "encode_gpu_h": round(DECODER_ENCODE_S / 3600, 2),
    "src": "decoder_ft RESULTS.txt (2000 steps 5908 s; encode 890 s); failed its dev gate",
}

# Data scale
out["derived"]["data_scale"] = {
    "local_clips": CLIPS_LOCAL, "local_frames": FRAMES_LOCAL,
    "share_of_stereocrafter_25M_frames": round(FRAMES_LOCAL / 25e6, 5),
    "share_of_stereoworld_11M_frames": round(FRAMES_LOCAL / 11e6, 5),
    "clips_vs_stereoworld_142520": round(CLIPS_LOCAL / 142520, 5),
    "clips_vs_stereopilot_60k_plus_48k": round(CLIPS_LOCAL / 108000, 5),
    "clips_vs_dcvideogen_417k": round(CLIPS_LOCAL / 417000, 6),
    "formula": "local / published",
}
# Interface share (from local_facts): 16-ch conv_in (16 noisy + 16 cond + 1 mask) + conv_out
out["derived"]["interface_16ch_share_of_unet_pct"] = round(141456 / (UNET_PARAMS_M * 1e6) * 100, 4)

with open(os.path.join(HERE, "COST_TABLE_v1.json"), "w") as f:
    json.dump(out, f, indent=1)
print(json.dumps(out["derived"], indent=1))
