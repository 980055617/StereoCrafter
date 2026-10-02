"""LENS-1 inference test: run the deployed inference path (inpainting_inference.run with the eval config
config/0160_overfit_inference_matched.json) but multiply the warped-frame cond latents by <scale>
(1.0 = deployed inference, 0.18215 = what the trainer feeds).

usage: python lens1_infer_scaled_cond.py <scale> <origin|/path/train_state.pt> <clip> <save_dir>
"""
import os, sys
ROOT = "/home/kawa/master_project/StereoCrafter"
sys.path.insert(0, ROOT)
os.chdir(ROOT)
os.environ.setdefault("MAMBA_SELF_ATTN_INCLUDE", "__nomatch__")
import torch
import inpainting_inference as ii

scale = float(sys.argv[1]); ck = sys.argv[2]; clip = sys.argv[3]; out = sys.argv[4]
extra = {}
if len(sys.argv) > 6:
    extra = dict(frames_chunk=int(sys.argv[5]), overlap=int(sys.argv[6]))
_orig = ii._Pipe._encode_vae_frames


def patched(self, *a, **k):
    lat = _orig(self, *a, **k)
    return lat * scale  # CFG zeros stay zeros


ii._Pipe._encode_vae_frames = patched
print(f"[lens1] cond-latent scale={scale} state={ck} clip={clip} extra={extra}", flush=True)
ii.run(config="config/0160_overfit_inference_matched.json",
       unet_state_path=(None if ck == "origin" else ck), expected_partial_unet_state=True,
       input_video_path=f"video_data/splatting/{clip}_splatting_results.mp4", save_dir=out, **extra)
print("[lens1] done", flush=True)
