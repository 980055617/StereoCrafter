#!/usr/bin/env python
"""[finalcheck_20261004/validate LOWMEM variant of infer_ll_hook_res.py: identical except that
inpainting_inference.read_and_prepare_video is rebound to lowmem_reader.read_and_prepare_video_lowmem.]
[finalcheck_20261004/validate COPY of scripts/distill/runs/skeptic1/infer_ll_hook.py.  ONLY change: optional
SK_H / SK_W env knobs passed to inpainting_inference.run as target_height / target_width together with
tile_num=1 (mirrors scripts/distill/fullhd.sh: --target_height --target_width --tile_num=1).  Unset -> the
config's 576x1024 and nothing extra is passed.]
SKEPTIC stacking runner: deployed inference with the LOSSLESS FFV1 writer, optionally with
(a) the shipped Mamba state loaded and (b) the beyond_distil student's 15 attn1 tensors swapped in.

Reuses, unmodified:
  scripts/distill/runs/fulldata_v2/beyond4/infer_lossless.py  -> the FFV1 writer monkeypatch
  scripts/distill/runs/diag_trainer/minift/xcheck_hybrid_minift.py  -> the tensor-swap hook idea
No tracked repo file is touched.

env:
  SK_CLIP     clip id (e.g. 0301)
  SK_OUT      save_dir (must be new)
  SK_STEPS    num_inference_steps (default 8)
  SK_GUID     min=max guidance (default 1.01)
  SK_UNET     unet_state_path to load (the Mamba deliverable) or "" -> None
  SK_CK       path to the distilled 15-tensor checkpoint, or "" -> no swap
  LOSSLESS_SBS=1 / KEEP_ANAGLYPH=0 consumed by infer_lossless
  MAMBA_SELF_ATTN_* consumed by blocks/mamba_diffusers_adapter
"""
import importlib.util, os, sys, collections

REPO = "/home/kawa/master_project/StereoCrafter"
sys.path.insert(0, REPO)
os.chdir(REPO)

# import the beyond4 lossless wrapper by path (applies II.write_video_opencv = FFV1 writer)
_spec = importlib.util.spec_from_file_location(
    "infer_lossless", f"{REPO}/scripts/distill/runs/fulldata_v2/beyond4/infer_lossless.py")
_IL = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_IL)

import torch
import inpainting_inference as ii

CLIP = os.environ["SK_CLIP"]
OUT = os.environ["SK_OUT"]
STEPS = int(os.environ.get("SK_STEPS", "8"))
GUID = float(os.environ.get("SK_GUID", "1.01"))
UNET = os.environ.get("SK_UNET", "").strip() or None
CK = os.environ.get("SK_CK", "").strip()
RES_H = os.environ.get("SK_H", "").strip()
RES_W = os.environ.get("SK_W", "").strip()
assert bool(RES_H) == bool(RES_W), "set both SK_H and SK_W or neither"

assert ii.write_video_opencv is _IL._patched_write, "lossless writer patch did not take"
# lowmem: rebind the reader name inside inpainting_inference (same tensors, chunked decode; see lowmem_reader.py)
sys.path.insert(0, f"{REPO}/scripts/distill/runs/finalcheck_20261004/validate")
from lowmem_reader import read_and_prepare_video_lowmem as _lowmem_read
ii.read_and_prepare_video = _lowmem_read
print("[sk] reader = lowmem_reader.read_and_prepare_video_lowmem", flush=True)
print(f"[sk] clip={CLIP} steps={STEPS} guid={GUID} unet={UNET} ck={CK or None} out={OUT} res={(RES_H + 'x' + RES_W) if RES_H else 'config'}", flush=True)

SWAP = {}
if CK:
    raw = torch.load(CK, map_location="cpu", weights_only=False)
    SWAP = raw.get("model", raw)
    print(f"[sk] distilled ckpt: {len(SWAP)} tensors", flush=True)

log = []
_call = ii._Pipe.__call__

def wrapped_call(self, *a, **k):
    unet = self.unet
    if SWAP and not getattr(unet, "_sk_installed", False):
        params = dict(unet.named_parameters())
        # the Mamba adapter re-parents the original attention as `<...>.attn1.origin_attn.*`
        mapping, missing = {}, []
        for key in SWAP:
            if key in params:
                mapping[key] = key
            else:
                alt = key.replace(".attn1.", ".attn1.origin_attn.")
                if alt in params:
                    mapping[key] = alt
                else:
                    missing.append(key)
        nplain = sum(1 for kk, vv in mapping.items() if kk == vv)
        nremap = len(mapping) - nplain
        print(f"[sk] swap mapping: {nplain} direct, {nremap} remapped to origin_attn.*, "
              f"{len(missing)} NOT FOUND", flush=True)
        if missing:
            print(f"[sk] missing keys: {missing[:3]} ...", flush=True)
        ndiff = 0
        with torch.no_grad():
            for kk, pk in mapping.items():
                p = params[pk]
                v = SWAP[kk].to(device=p.device, dtype=p.dtype)
                ndiff += int((v != p).any())
                p.copy_(v)
        print(f"[sk] tensors that differed from the loaded model: {ndiff}/{len(mapping)}", flush=True)
        # report whether the modules we just wrote to are even evaluated
        from blocks.mamba_diffusers_adapter import GatedResidualMambaSelfAttention as G
        gated = [(n, float(m.mamba_gate), bool(m.reference_disabled))
                 for n, m in unet.named_modules() if isinstance(m, G)]
        for n, g, rd in gated:
            print(f"[sk] gated module {n}: gate={g} reference_disabled={rd} "
                  f"origin_attn_evaluated={not (rd or g >= 1.0)}", flush=True)
        unet._sk_installed = True
    return _call(self, *a, **k)

ii._Pipe.__call__ = wrapped_call

kw = dict(config="config/0160_overfit_inference_matched.json",
          input_video_path=f"video_data/splatting/{CLIP}_splatting_results.mp4",
          save_dir=OUT, num_inference_steps=STEPS,
          min_guidance_scale=GUID, max_guidance_scale=GUID)
if RES_H:
    kw["target_height"] = int(RES_H)
    kw["target_width"] = int(RES_W)
    kw["tile_num"] = 1
if UNET is None:
    kw["unet_state_path"] = None
else:
    kw["unet_state_path"] = UNET
    kw["expected_partial_unet_state"] = True
    kw["mamba_gate_override"] = 1.0
ii.run(**kw)
print("[sk] done", flush=True)
