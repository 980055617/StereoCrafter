"""Shared harness for the mech/ experiments (Test B = per-step x0-hat, Test A = on-trajectory fine-tune).

Never edits pipelines/*: the deployed sampler is *wrapped* at two points
  - pipe.scheduler.step   -> stashes (sigma, unscaled sample y_k, out.pred_original_sample)
  - pipe.unet.forward     -> stashes the literal call args (9-ch input AFTER scale_model_input+concat, t,
                             encoder_hidden_states, added_time_ids) and the raw v output
so the captured trajectory is exactly the deployed one (including scheduler.step's per-step randn_tensor draw,
which consumes RNG even at gamma=0).

Deployed config mirrored from config/0160_overfit_inference_matched.json:
  frames_chunk 14, overlap 3 (=> stride 11), tile_num 1, 8 steps, bf16, decode_chunk_size 2,
  guidance 1.01 (>1.0 => CFG IS ACTIVE, UNet batch 2: [uncond(zeros), cond]), noise_aug 0.0,
  576x1024 centre crop of the //128 crop, noise_seed 1234.
"""
import os, sys, math, json

REPO = "/home/kawa/master_project/StereoCrafter"
os.environ.setdefault("MAMBA_SELF_ATTN_INCLUDE", "__nomatch__")   # no Mamba: pure origin UNet
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

import torch
import torch.nn.functional as F
import numpy as np
from decord import VideoReader, cpu

CLIP = "0301"
NF = 14
STRIDE = 11
SPLAT = f"video_data/splatting/{CLIP}_splatting_results.mp4"
TRAINMP4 = f"video_data/train/{CLIP}_train.mp4"
PRE = "weights/stable-video-diffusion-img2vid-xt-1-1/"
UNET_PATH = "weights/StereoCrafter/"
PREF = ["up_blocks.3.attentions.0.transformer_blocks.0.attn1.",
        "up_blocks.3.attentions.1.transformer_blocks.0.attn1.",
        "up_blocks.3.attentions.2.transformer_blocks.0.attn1."]
GUID = 1.01
STEPS8 = 8
DECODE_CHUNK = 2


def new_out(base_rel, sub):
    """Never overwrite: append _2, _3 ... if the directory already exists."""
    out = os.path.join(base_rel, sub)
    k = 1
    while os.path.exists(out):
        k += 1
        out = os.path.join(base_rel, f"{sub}_{k}")
    os.makedirs(out)
    return out


# ---------------------------------------------------------------- data
def read_deployed_inputs():
    """utils/inpainting.read_and_prepare_video on the SPLATTING video + inpainting_inference._center_crop_frames.
    Returns (fps, left, warped_cond, mask) each [T,C,H,W] float in [0,1] at 576x1024."""
    from utils.inpainting import read_and_prepare_video
    from inpainting_inference import _center_crop_frames
    fps, left, warped, mask = read_and_prepare_video(SPLAT)
    left = _center_crop_frames(left, 576, 1024)
    warped = _center_crop_frames(warped, 576, 1024)
    mask = _center_crop_frames(mask, 576, 1024)
    return fps, left, warped, mask


def crop_quadrants(fr):
    """xcheck_mini_ft.crop_quadrants, verbatim: fr [T,3,H2,W2] 2x2 tile -> (BR cond, BL mask 1ch, TR gt, TL left)
    at the registered 576x1024 crop."""
    H, W = fr.shape[2] // 2, fr.shape[3] // 2
    TL, TR, BL, BR = fr[:, :, :H, :W], fr[:, :, :H, W:], fr[:, :, H:, :W], fr[:, :, H:, W:]
    h, w = H // 128 * 128, W // 128 * 128
    top, left = (h - 576) // 2, (w - 1024) // 2
    sl = (slice(None), slice(None), slice(top, top + 576), slice(left, left + 1024))
    return BR[:, :, :h, :w][sl], BL[:, :, :h, :w][sl].mean(dim=1, keepdim=True), TR[:, :, :h, :w][sl], TL[:, :, :h, :w][sl]


def read_train_tile(s, e):
    vr = VideoReader(TRAINMP4, ctx=cpu(0))
    return torch.from_numpy(vr.get_batch(list(range(s, e))).asnumpy()).permute(0, 3, 1, 2).float() / 255.0


def window_starts(n_frames):
    return list(range(0, n_frames - NF + 1, STRIDE))


# ---------------------------------------------------------------- metrics
def sharp01(x):
    """score_clip.py line 36 verbatim, on a [T,3,H,W] tensor in [0,1]: mean |horizontal first difference|."""
    return float((x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean())


# ---------------------------------------------------------------- model
def build_pipe(dt=torch.bfloat16, dev="cuda:0"):
    """Deployed construction (inpainting_inference.main lines 92-145): same classes, same dtype, VAE helpers on.
    Importing inpainting_inference also runs apply_mamba_time_patch() as deployment does."""
    from transformers import CLIPVisionModelWithProjection
    from diffusers.models.unets.unet_spatio_temporal_condition import UNetSpatioTemporalConditionModel
    from diffusers.models.autoencoders.autoencoder_kl_temporal_decoder import AutoencoderKLTemporalDecoder
    import inpainting_inference as ii
    from utils.training_pipeline import enable_vae_memory_helpers

    image_encoder = CLIPVisionModelWithProjection.from_pretrained(PRE, subfolder="image_encoder", variant="fp16", torch_dtype=dt)
    vae = AutoencoderKLTemporalDecoder.from_pretrained(PRE, subfolder="vae", variant="fp16", torch_dtype=dt)
    unet = UNetSpatioTemporalConditionModel.from_pretrained(UNET_PATH, subfolder="unet_diffusers", low_cpu_mem_usage=True, torch_dtype=dt)
    image_encoder.requires_grad_(False); vae.requires_grad_(False); unet.requires_grad_(False)
    pipe = ii._Pipe.from_pretrained(PRE, image_encoder=image_encoder, vae=vae, unet=unet, torch_dtype=dt)
    enable_vae_memory_helpers(pipe)
    pipe = pipe.to(dev)
    pipe.vae.eval(); pipe.image_encoder.eval(); pipe.unet.eval()
    return pipe


def trainable_15(unet):
    t = [(n, p) for n, p in unet.named_parameters() if any(n.startswith(q) for q in PREF) and "origin_attn" not in n]
    assert len(t) == 15, [n for n, _ in t]
    return t


def load_swap(unet, ck_path):
    """Permanently copy a minift checkpoint's 15 tensors into the UNet (the 'all steps' hybrid mode)."""
    raw = torch.load(ck_path, map_location="cpu", weights_only=False)
    m = raw.get("model", raw) if isinstance(raw, dict) else raw
    sel = {k: v for k, v in m.items() if any(k.startswith(p) for p in PREF) and "origin_attn" not in k}
    params = dict(unet.named_parameters())
    n_diff = 0
    with torch.no_grad():
        for k, v in sel.items():
            tgt = params[k]
            vv = v.to(device=tgt.device, dtype=tgt.dtype)
            n_diff += int((vv != tgt).any())
            tgt.copy_(vv)
    return len(sel), n_diff


# ---------------------------------------------------------------- sampler capture
def run_window(pipe, cond, mask, seed=1234, steps=STEPS8, guid=GUID, keep_unet_in=False):
    """Run the deployed sampler on ONE 14-frame window and capture every step.

    cond, mask: [14,3,576,1024] / [14,1,576,1024] in [0,1]
    Returns (final_latents [1,14,4,72,128] on cpu float32, rec dict).
    rec keys: sigma[8], t[8], y_raw[8] (unscaled latents fed to scheduler.step),
              x0hat[8] (out.pred_original_sample), unet_in[8] (bf16 [2,14,9,72,128], only if keep_unet_in),
              v_cfg_cond[8] (raw UNet output, cond half), v_cfg_uncond[8], emb[2,1,1024], add[2,3], init_lat_stats
    """
    from utils.inpainting import spatial_tiled_process
    sch = pipe.scheduler
    rec = {"sigma": [], "t": [], "y_raw": [], "x0hat": [], "unet_in": [], "v_cfg_cond": [], "v_cfg_uncond": []}

    orig_step = sch.step

    def step_hook(model_output, timestep, sample, **kw):
        sig = float(sch.sigmas[sch.step_index])
        out = orig_step(model_output, timestep, sample, **kw)
        rec["sigma"].append(sig)
        rec["y_raw"].append(sample.detach().to(torch.float32).cpu())
        rec["x0hat"].append(out.pred_original_sample.detach().to(torch.float32).cpu())
        return out

    _fwd = pipe.unet.forward

    def fwd(sample, timestep, *a, **k):
        out = _fwd(sample, timestep, *a, **k)
        v = out[0] if isinstance(out, (tuple, list)) else out.sample
        rec["t"].append(float(timestep.flatten()[0]) if torch.is_tensor(timestep) else float(timestep))
        if keep_unet_in:
            rec["unet_in"].append(sample.detach().cpu().clone())
        rec["v_cfg_uncond"].append(v[0:1].detach().cpu().clone())
        rec["v_cfg_cond"].append(v[1:2].detach().cpu().clone())
        ehs = k.get("encoder_hidden_states", a[0] if a else None)
        ati = k.get("added_time_ids", None)
        rec["emb"] = ehs.detach().cpu().clone()
        rec["add"] = ati.detach().cpu().clone()
        return out

    sch.step = step_hook
    pipe.unet.forward = fwd
    try:
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        lat = spatial_tiled_process(cond, mask, pipe, 1, spatial_n_compress=8,
                                    min_guidance_scale=float(guid), max_guidance_scale=float(guid),
                                    decode_chunk_size=DECODE_CHUNK, fps=7, motion_bucket_id=127,
                                    noise_aug_strength=0.0, num_inference_steps=steps, generator=None)
    finally:
        sch.step = orig_step
        pipe.unet.forward = _fwd
    rec["init_lat_stats"] = (float(rec["y_raw"][0].mean()), float(rec["y_raw"][0].std()), float(rec["y_raw"][0].abs().sum()))
    return lat.unsqueeze(0).detach().to(torch.float32).cpu(), rec


@torch.no_grad()
def decode01(pipe, lat, num_frames=NF, chunk=DECODE_CHUNK):
    """pipeline.decode_latents + VaeImageProcessor.denormalize ((x/2+0.5).clamp(0,1)), as deployment does.
    Returns (img01 [T,3,H,W] in [0,1], out_of_range_fraction before clamping)."""
    fr = pipe.decode_latents(lat.to(pipe.vae.dtype).to(pipe.device), num_frames=num_frames, decode_chunk_size=chunk)
    x = fr[0].permute(1, 0, 2, 3).float()          # [T,3,H,W], roughly [-1,1]
    raw = x / 2 + 0.5
    oor = float(((raw < 0) | (raw > 1)).float().mean())
    return raw.clamp(0, 1).cpu(), oor
