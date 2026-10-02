"""../minift/xcheck_hybrid_minift.py with the sampler step count overridable (env MECH_STEPS, default 8).
Used for the 1-step tiebreaker: if a trained model "learned to jump", its 1-step output should BEAT origin's.
usage: MINIFT_CK=<ck.pt> MECH_STEPS=1 python hybrid_nsteps.py <originall|e1all> <clip> <save_dir>
"""
import os, sys, json
ROOT = "/home/kawa/master_project/StereoCrafter"; sys.path.insert(0, ROOT); os.chdir(ROOT)
os.environ.setdefault("MAMBA_SELF_ATTN_INCLUDE", "__nomatch__")
import torch
import inpainting_inference as ii
mode, clip, out = sys.argv[1], sys.argv[2], sys.argv[3]
NSTEPS = int(os.environ.get("MECH_STEPS", "8"))
CK = os.environ["MINIFT_CK"]
PREF = ["up_blocks.3.attentions.0.transformer_blocks.0.attn1.", "up_blocks.3.attentions.1.transformer_blocks.0.attn1.",
        "up_blocks.3.attentions.2.transformer_blocks.0.attn1."]
raw = torch.load(CK, map_location="cpu", weights_only=False); m = raw.get("model", raw)
e1 = {k: v for k, v in m.items() if any(k.startswith(p) for p in PREF) and "origin_attn" not in k}
print(f"[hybrid] mode={mode} steps={NSTEPS} swap-set={len(e1)} tensors", flush=True)
_call = ii._Pipe.__call__
def wrapped_call(self, *a, **k):
    unet = self.unet
    if not getattr(unet, "_hybrid_installed", False):
        params = dict(unet.named_parameters())
        origin = {kk: params[kk].detach().clone() for kk in e1}
        e1dev = {kk: e1[kk].to(device=params[kk].device, dtype=params[kk].dtype) for kk in e1}
        nd = sum(int((e1dev[kk] != origin[kk]).any()) for kk in e1)
        print(f"[hybrid] tensors differing from origin in swap-set: {nd}/{len(e1)}", flush=True)
        src = e1dev if mode != "originall" else origin
        with torch.no_grad():
            for kk in e1: params[kk].copy_(src[kk])
        unet._hybrid_installed = True
    return _call(self, *a, **k)
ii._Pipe.__call__ = wrapped_call
ii.run(config="config/0160_overfit_inference_matched.json", unet_state_path=None, num_inference_steps=NSTEPS,
       input_video_path=f"video_data/splatting/{clip}_splatting_results.mp4", save_dir=out)
print("[hybrid] done", flush=True)
