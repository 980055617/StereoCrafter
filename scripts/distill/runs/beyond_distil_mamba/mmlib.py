"""MAMBA-side beyond-distil harness: the same 25-step-density trajectory objective, but the model whose
parameters move is the SHIPPED 5-slot light-Mamba deliverable instead of the origin attention.

WHY A NEW LIBRARY AT ALL.  beyond_distil/bdlib.py -> mech/mechlib.py forces the ORIGIN UNet by doing
    os.environ.setdefault("MAMBA_SELF_ATTN_INCLUDE", "__nomatch__")
at import time.  Because that is a *setdefault*, this module only has to publish the deployed Mamba env
BEFORE mechlib is imported and the very same harness builds the Mamba UNet.  Every script in this
directory therefore imports `mmlib` first, and mmlib imports bdlib/mechlib for it.

THE DEPLOYED MAMBA CONSTRUCTION, mirrored from inpainting_inference.main lines 110-226:
    1. same classes / dtype / VAE helpers as mechlib.build_pipe
    2. _Pipe.__init__ swaps attn1 for GatedResidualMambaSelfAttention in the 5 slots named by
       MAMBA_SELF_ATTN_INCLUDE (d_state 128, expand 1, fwd-only, gated_residual)
    3. materialize_mamba_time_embed_proj_from_state_dict  <-- MANDATORY.  time_embed_proj is created
       LAZILY AND ZERO-INITIALISED on first forward, and a zero FiLM is the identity, so if the 6
       time_embed_proj tensors were silently dropped as "unexpected" the model would still run, still
       pass every algebra/M=1/input-reconstruction control, and be a DIFFERENT, WEAKER model than the
       one that was shipped.  verify_mamba_v1.py asserts they are present and nonzero.
    4. unet -> fp32, load_state_dict(strict=False), unet -> bf16   (the deployed rounding order)
    5. set_gated_mamba_gate(1.0): forward() returns the Mamba output and origin_attn is never evaluated

TRAINABLE SET.  Per slot the module holds
    attn1.fwd.core.*           393,935  evaluated (MAMBA_BIDIRECTIONAL_MODE=fwd)
    attn1.time_embed_proj.*    819,840  evaluated (FiLM on the UNet time embedding)
    attn1.bwd.core.*           393,935  NEVER evaluated in fwd mode      -> excluded
    attn1.origin_attn.*        409,920  NEVER evaluated at gate 1.0      -> excluded
    attn1.mamba_gate           buffer, not a parameter                  -> excluded
so 10 tensors / 1,213,775 params per slot.
"""
import os
import sys

REPO = "/home/kawa/master_project/StereoCrafter"

# ---- the deployed Mamba env, PUBLISHED BEFORE mechlib's setdefault can force the origin UNet -------
os.environ.setdefault("MAMBA_SELF_ATTN_INCLUDE", "down_blocks.0.*,up_blocks.3.*")
os.environ.setdefault("MAMBA_SELF_ATTN_EXCLUDE", "__nomatch__")
os.environ.setdefault("MAMBA_SELF_ATTN_D_STATE", "128")
os.environ.setdefault("MAMBA_SELF_ATTN_EXPAND", "1")
os.environ.setdefault("MAMBA_BIDIRECTIONAL_MODE", "fwd")
os.environ.setdefault("MAMBA_SELF_ATTN_REPLACEMENT", "gated_residual")

sys.path.insert(0, os.path.join(REPO, "scripts/distill/runs/beyond_distil_mamba"))
import bdlib as B                                            # noqa: E402  (imports mechlib, chdirs to REPO)
import mechlib as M                                          # noqa: E402
import torch                                                 # noqa: E402

MAMBA_CK = "/mnt/ssd_data/stereocrafter_weights/_distill_injected/light_lvl0_fulldata333_v2_8k_mamba_only.pt"
MAMBA_MD5 = "e9c232878319d041680e7fb3be74bf10"

SLOTS_DOWN0 = ["down_blocks.0.attentions.0", "down_blocks.0.attentions.1"]
SLOTS_UP3 = ["up_blocks.3.attentions.0", "up_blocks.3.attentions.1", "up_blocks.3.attentions.2"]
SLOTS_ALL5 = SLOTS_DOWN0 + SLOTS_UP3

TRAIN_TAILS = ("fwd.core.", "time_embed_proj.")          # the only tensors the deployed forward evaluates


def prefixes(slots):
    return [f"{s}.transformer_blocks.0.attn1." for s in slots]


PREF_UP3 = prefixes(SLOTS_UP3)
PREF_ALL5 = prefixes(SLOTS_ALL5)
PREF_DOWN0 = prefixes(SLOTS_DOWN0)


def slot_set(name):
    name = (name or "up3").strip().lower()
    return {"up3": PREF_UP3, "all5": PREF_ALL5, "down0": PREF_DOWN0}[name]


def is_trainable_name(n, prefs):
    """A parameter is trainable iff it is inside one of the chosen attn1 slots AND on the evaluated path."""
    if not any(n.startswith(p) for p in prefs):
        return False
    if "origin_attn" in n:
        return False
    tail = n.split(".attn1.", 1)[1]
    return tail.startswith(TRAIN_TAILS)


def trainable_mamba(unet, prefs):
    return [(n, p) for n, p in unet.named_parameters() if is_trainable_name(n, prefs)]


# ---------------------------------------------------------------- the deployed Mamba pipe
def build_pipe_mamba(dt=torch.bfloat16, dev="cuda:0", ck=MAMBA_CK, gate=1.0, verbose=True):
    """inpainting_inference.main's construction + state load, with nothing else changed.
    Returns (pipe, info) where info carries every fidelity number verify_mamba_v1.py asserts on."""
    from transformers import CLIPVisionModelWithProjection
    from diffusers.models.unets.unet_spatio_temporal_condition import UNetSpatioTemporalConditionModel
    from diffusers.models.autoencoders.autoencoder_kl_temporal_decoder import AutoencoderKLTemporalDecoder
    import inpainting_inference as ii                         # also runs apply_mamba_time_patch()
    from utils.training_pipeline import enable_vae_memory_helpers
    from blocks.mamba_diffusers_adapter import (
        GatedResidualMambaSelfAttention as G,
        materialize_mamba_time_embed_proj_from_state_dict,
        set_gated_mamba_gate,
    )

    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        M.PRE, subfolder="image_encoder", variant="fp16", torch_dtype=dt)
    vae = AutoencoderKLTemporalDecoder.from_pretrained(M.PRE, subfolder="vae", variant="fp16", torch_dtype=dt)
    unet = UNetSpatioTemporalConditionModel.from_pretrained(
        M.UNET_PATH, subfolder="unet_diffusers", low_cpu_mem_usage=True, torch_dtype=dt)
    image_encoder.requires_grad_(False); vae.requires_grad_(False); unet.requires_grad_(False)
    pipe = ii._Pipe.from_pretrained(M.PRE, image_encoder=image_encoder, vae=vae, unet=unet, torch_dtype=dt)
    enable_vae_memory_helpers(pipe)

    gated = [(n, m) for n, m in pipe.unet.named_modules() if isinstance(m, G)]
    info = dict(n_gated=len(gated), gated_names=[n for n, _ in gated], ck=ck)

    raw = torch.load(ck, map_location="cpu", weights_only=False)
    sd = raw.get("model", raw) if isinstance(raw, dict) else raw
    info["ck_tensors"] = len(sd)
    info["materialized"] = materialize_mamba_time_embed_proj_from_state_dict(pipe.unet, sd)
    pipe.unet.to(dtype=torch.float32)
    missing, unexpected = pipe.unet.load_state_dict(sd, strict=False)
    pipe.unet.to(dtype=dt)
    info["n_missing"] = len(missing)
    info["n_unexpected"] = len(unexpected)
    info["unexpected"] = list(unexpected)[:8]
    info["n_gate_updated"] = set_gated_mamba_gate(pipe.unet, float(gate))

    # bit-fidelity of every checkpoint tensor AFTER the fp32 -> bf16 round trip
    ps = dict(pipe.unet.named_parameters()); bs = dict(pipe.unet.named_buffers())
    bad, absent = [], []
    for k, v in sd.items():
        t = ps.get(k, bs.get(k))
        if t is None:
            absent.append(k); continue
        if not torch.equal(t.detach().cpu(), v.to(t.dtype).cpu()):
            bad.append(k)
    info["ck_absent"] = absent
    info["ck_mismatch"] = bad

    pipe = pipe.to(dev)
    pipe.vae.eval(); pipe.image_encoder.eval(); pipe.unet.eval()
    info["gates"] = [(n, float(m.mamba_gate), bool(m.reference_disabled)) for n, m in gated]
    info["film"] = {n: (m.time_embed_proj is not None,
                        float(m.time_embed_proj.weight.abs().max()) if m.time_embed_proj is not None else -1.0)
                    for n, m in gated}
    if verbose:
        print(f"[mmlib] gated modules {info['n_gated']}  materialized {info['materialized']}  "
              f"missing {info['n_missing']}  unexpected {info['n_unexpected']}  gate_updated {info['n_gate_updated']}",
              flush=True)
        print(f"[mmlib] ck tensors {info['ck_tensors']}  absent {len(absent)}  bf16-mismatch {len(bad)}", flush=True)
    return pipe, info


def load_swap_mamba(unet, ck_path, prefs=None):
    """Copy a beyond_distil_mamba checkpoint's tensors into the live UNet (the 'all steps' student mode)."""
    raw = torch.load(ck_path, map_location="cpu", weights_only=False)
    m = raw.get("model", raw) if isinstance(raw, dict) else raw
    params = dict(unet.named_parameters())
    n_diff = 0
    with torch.no_grad():
        for k, v in m.items():
            if prefs is not None and not any(k.startswith(p) for p in prefs):
                continue
            t = params[k]
            vv = v.to(device=t.device, dtype=t.dtype)
            n_diff += int((vv != t).any())
            t.copy_(vv)
    return len(m), n_diff


# ---------------------------------------------------------------- deployed-faithful decode (for the pixel control)
@torch.no_grad()
def decode_deployed_uint8(pipe, lat, num_frames=M.NF, chunk=M.DECODE_CHUNK):
    """inpainting_inference lines 318-330 verbatim: decode_latents -> tensor2vid(output_type='pil') -> uint8."""
    import numpy as np
    from pipelines.mamba_stereo_video_inpainting_pipeline import tensor2vid
    vf = pipe.decode_latents(lat.to(pipe.vae.dtype).to(pipe.device),
                             num_frames=num_frames, decode_chunk_size=chunk)
    imgs = tensor2vid(vf, pipe.image_processor, output_type="pil")[0]
    return np.stack([np.array(im) for im in imgs])          # [T,H,W,3] uint8 RGB
