"""regB copy of xcheck_hybrid_regB_ll.py with an extra env knob MINIFT_STEPS (sampler steps, default 8; 25 reproduces the s25 config).
Original header: regB copy of scripts/distill/runs/diag_trainer/minift/xcheck_hybrid_minift.py (checkpoint from env MINIFT_CK) with ONE change:
   the _sbs output is written LOSSLESSLY (FFV1) by importing scripts/distill/runs/fulldata_v2/beyond4/infer_lossless.py, which
   rebinds inpainting_inference.write_video_opencv (LOSSLESS_SBS=1 default; writer_md5.txt records the pre-encode array md5).
   The sampling path is untouched: mode originall must reproduce the pre-encode md5 of outputs/beyond4_lossless/clips/0301_origin_ll
   (2e533d7755c950d2fc95043f6fb0a51d) byte for byte.
Cross-check: deployed inference (eval config, 14f, cond x1.0) with the fine-tuned attn1 tensors swapped in
   only for part of the sampler / part of the slots.  mode:
     originall  : origin everywhere (control)
     e1all      : the checkpoint's tensors everywhere
     e1high     : e001 tensors only when t=0.25*ln(sigma) > 1.0  (sigma 700/287/103 = first 3 of 8 steps), origin otherwise
     e1low      : e001 tensors only when t <= 1.0 (sigma 31/7.3/1.17/0.097/0.002 = last 5 steps), origin otherwise
     e1up3      : e001 tensors only in the 3 up_blocks.3 slots (15 tensors), all steps
     e1down0    : e001 tensors only in the 2 down_blocks.0 slots (10 tensors), all steps
     e1range:lo:hi : e001 tensors only when lo < t <= hi  (t = 0.25 ln sigma; steps: 1.638 1.415 1.158 0.858 0.496 0.039 -0.582 -1.554)
   usage: python xcheck_hybrid.py <mode> <clip> <save_dir>
"""
import os, sys, json
ROOT = "/home/kawa/master_project/StereoCrafter"; sys.path.insert(0, ROOT); os.chdir(ROOT)
os.environ.setdefault("MAMBA_SELF_ATTN_INCLUDE", "__nomatch__")
import torch
sys.path.insert(0, os.path.join(ROOT, "scripts/distill/runs/fulldata_v2/beyond4"))
import infer_lossless as IL          # rebinds inpainting_inference.write_video_opencv -> FFV1 writer at import
import inpainting_inference as ii
assert ii.write_video_opencv is IL._patched_write, "lossless writer not installed"
print(f"[lossless] LOSSLESS_SBS={int(IL.LOSSLESS_SBS)} KEEP_ANAGLYPH={int(IL.KEEP_ANAGLYPH)} ffmpeg={IL.FFMPEG}", flush=True)
mode, clip, out = sys.argv[1], sys.argv[2], sys.argv[3]
assert not os.path.exists(out), f"refusing to overwrite existing output dir {out}"
CK = os.environ["MINIFT_CK"]   # env override: path to a state-dict (or {"model": sd}) holding the tensors to swap in
PREF = ["down_blocks.0.attentions.0.transformer_blocks.0.attn1.", "down_blocks.0.attentions.1.transformer_blocks.0.attn1.",
        "up_blocks.3.attentions.0.transformer_blocks.0.attn1.", "up_blocks.3.attentions.1.transformer_blocks.0.attn1.",
        "up_blocks.3.attentions.2.transformer_blocks.0.attn1."]
raw = torch.load(CK, map_location="cpu", weights_only=False); m = raw.get("model", raw)
e1 = {k: v for k, v in m.items() if any(k.startswith(p) for p in PREF) and "origin_attn" not in k}
if mode == "e1up3": e1 = {k: v for k, v in e1.items() if k.startswith("up_blocks.3")}
if mode == "e1down0": e1 = {k: v for k, v in e1.items() if k.startswith("down_blocks.0")}
print(f"[hybrid] mode={mode} swap-set={len(e1)} tensors", flush=True)
T_SPLIT = 1.0
log = []
_call = ii._Pipe.__call__
def wrapped_call(self, *a, **k):
    unet = self.unet
    if not getattr(unet, "_hybrid_installed", False):
        params = dict(unet.named_parameters())
        origin = {kk: params[kk].detach().clone() for kk in e1}
        e1dev = {kk: e1[kk].to(device=params[kk].device, dtype=params[kk].dtype) for kk in e1}
        nd = sum(int((e1dev[kk] != origin[kk]).any()) for kk in e1)
        print(f"[hybrid] tensors differing from origin in swap-set: {nd}/{len(e1)}", flush=True)
        st = {"cur": None}
        def use(which):
            if st["cur"] == which: return
            src = e1dev if which == "e1" else origin
            with torch.no_grad():
                for kk in e1: params[kk].copy_(src[kk])
            st["cur"] = which
        _fwd = unet.forward
        def fwd(sample, timestep, *fa, **fk):
            t = float(timestep.flatten()[0]) if torch.is_tensor(timestep) else float(timestep)
            high = t > T_SPLIT
            if mode.startswith("e1range:"):
                lo, hi = (float(x) for x in mode.split(":")[1:3]); use("e1" if (lo < t <= hi) else "origin")
            elif mode == "e1high": use("e1" if high else "origin")
            elif mode == "e1low": use("origin" if high else "e1")
            elif mode == "originall": use("origin")
            else: use("e1")
            log.append((round(t, 4), st["cur"]))
            return _fwd(sample, timestep, *fa, **fk)
        unet.forward = fwd; unet._hybrid_installed = True
    return _call(self, *a, **k)
ii._Pipe.__call__ = wrapped_call
NSTEPS = int(os.environ.get("MINIFT_STEPS", "8"))   # regB extra: sampler step count override (25 == the origin+s25 ceiling runs of beyond4)
print(f"[hybrid] num_inference_steps={NSTEPS}", flush=True)
ii.run(config="config/0160_overfit_inference_matched.json", unet_state_path=None, num_inference_steps=NSTEPS,
       input_video_path=f"video_data/splatting/{clip}_splatting_results.mp4", save_dir=out)
import collections
print("[hybrid] (t, weights) usage:", sorted(collections.Counter(log).items()), flush=True)
print("[hybrid] done", flush=True)
