#!/usr/bin/env python
"""What does option (B) actually give up?  UNet-MODULE attn1 time for origin / Mamba 5-slot / Mamba 2-slot.

The A-vs-B decision cannot be made on wall clock at 576x1024 (origin 165-195 s, 5-slot 173-201 s, 2-slot
179-210 s -- all inside the noise).  The published speed wins are UNet-MODULE times, produced by
inpainting_inference.py's own --module_profile_json / --module_profile_include (default *.attn1).  This
runs that same instrument over the three slot configurations at one resolution, for a fixed number of
chunks, so the three rows are directly comparable.

env: PM_LABEL  PM_SLOTS=none|down0|all5  PM_H  PM_W  PM_CHUNKS  PM_OUT
"""
import os, sys

REPO = "/home/kawa/master_project/StereoCrafter"
LAB = os.environ["PM_LABEL"]
SLOTS = os.environ.get("PM_SLOTS", "all5")
H = int(os.environ.get("PM_H", "576")); W = int(os.environ.get("PM_W", "1024"))
CHUNKS = int(os.environ.get("PM_CHUNKS", "2"))
OUTD = os.environ["PM_OUT"]
CLIP = os.environ.get("PM_CLIP", "0301")

INC = {"none": "__nomatch__", "down0": "down_blocks.0.*", "all5": "down_blocks.0.*,up_blocks.3.*"}[SLOTS]
os.environ["MAMBA_SELF_ATTN_INCLUDE"] = INC
os.environ["MAMBA_SELF_ATTN_EXCLUDE"] = "__nomatch__"
os.environ["MAMBA_SELF_ATTN_D_STATE"] = "128"
os.environ["MAMBA_SELF_ATTN_EXPAND"] = "1"
os.environ["MAMBA_BIDIRECTIONAL_MODE"] = "fwd"
os.environ["MAMBA_SELF_ATTN_REPLACEMENT"] = "gated_residual"
sys.path.insert(0, REPO)
os.chdir(REPO)
os.makedirs(OUTD, exist_ok=True)

import inpainting_inference as ii

kw = dict(config="config/0160_overfit_inference_matched.json",
          input_video_path=f"video_data/splatting/{CLIP}_splatting_results.mp4",
          save_dir=OUTD, num_inference_steps=8, min_guidance_scale=1.01, max_guidance_scale=1.01,
          target_height=H, target_width=W, max_profile_chunks=CHUNKS,
          module_profile_json=os.path.join(OUTD, f"module_profile_{LAB}.json"),
          module_profile_include="*.attn1")
if SLOTS == "none":
    kw["unet_state_path"] = None
else:
    kw.update(unet_state_path="/mnt/ssd_data/stereocrafter_weights/_distill_injected/"
                              "light_lvl0_fulldata333_v2_8k_mamba_only.pt",
              expected_partial_unet_state=True, mamba_gate_override=1.0)
print(f"[pm] label={LAB} slots={SLOTS} include={INC} {H}x{W} chunks={CHUNKS}", flush=True)
ii.run(**kw)
print("PM_DONE", flush=True)
