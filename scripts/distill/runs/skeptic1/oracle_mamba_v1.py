#!/usr/bin/env python
"""GATE for the follow-up experiment: is the s25-equivalent headroom also present INSIDE the shipped
Mamba model, reachable by fixing only coarse steps k=4,5,6?

Same construction as beyond_distil/oracle_infer.py (its fine_step, its latent substitution), but the
UNet is the shipped Mamba deliverable and the output is written LOSSLESSLY so the row is comparable to
my mamba_ll / mamba_s25_ll rows.  Nothing is trained.

env: SK_CLIP  SK_OUT  BD_M (default 4)  BD_SUBST (default "4,5,6")
     MAMBA_SELF_ATTN_* as for any Mamba run;  SK_UNET = the Mamba state.
"""
import importlib.util, os, sys, json, collections

REPO = "/home/kawa/master_project/StereoCrafter"
sys.path.insert(0, REPO); sys.path.insert(0, f"{REPO}/scripts/distill/runs/beyond_distil")
os.chdir(REPO)

_spec = importlib.util.spec_from_file_location(
    "infer_lossless", f"{REPO}/scripts/distill/runs/fulldata_v2/beyond4/infer_lossless.py")
_IL = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_IL)

import bdlib as B
import torch
import inpainting_inference as ii

CLIP = os.environ["SK_CLIP"]; SAVE = os.environ["SK_OUT"]
MM = int(os.environ.get("BD_M", "4"))
SS = os.environ.get("BD_SUBST", "4,5,6")
SUBST = set(range(8)) if SS == "all" else {int(x) for x in SS.split(",")}
UNET = os.environ.get("SK_UNET", "").strip() or None
assert ii.write_video_opencv is _IL._patched_write, "lossless writer patch did not take"
print(f"[oraclem] clip {CLIP} M={MM} subst {sorted(SUBST)} unet={UNET}", flush=True)

stash = {}; used = collections.Counter()
_call = ii._Pipe.__call__

def wrapped_call(self, *a, **k):
    if not getattr(self.unet, "_bd_installed", False):
        unet = self.unet; sch = self.scheduler
        _fwd = unet.forward
        def fwd(sample, timestep, *fa, **fk):
            stash["frame"] = sample[:, :, 4:8]; stash["mask"] = sample[:, :, 8:9]
            stash["emb"] = fk.get("encoder_hidden_states", fa[0] if fa else None)
            stash["add"] = fk.get("added_time_ids", None)
            return _fwd(sample, timestep, *fa, **fk)
        _step = sch.step
        def step(model_output, timestep, sample, **kw):
            kk = sch.step_index if sch.step_index is not None else 0
            sg = float(sch.sigmas[kk]); sg1 = float(sch.sigmas[kk + 1])
            out = _step(model_output, timestep, sample, **kw)
            if kk not in SUBST:
                used[(kk, "coarse")] += 1; return out
            xt = B.fine_step(unet, sample.float(), sg, sg1, MM, stash["frame"], stash["mask"],
                             stash["emb"], stash["add"], guid=B.GUID, v0=model_output.float(),
                             dt=model_output.dtype)
            used[(kk, "fine")] += 1
            out.prev_sample = xt.to(model_output.dtype)
            return out
        unet.forward = fwd; sch.step = step; unet._bd_installed = True
    return _call(self, *a, **k)

ii._Pipe.__call__ = wrapped_call
kw = dict(config="config/0160_overfit_inference_matched.json",
          input_video_path=f"video_data/splatting/{CLIP}_splatting_results.mp4", save_dir=SAVE)
if UNET is None:
    kw["unet_state_path"] = None
else:
    kw.update(unet_state_path=UNET, expected_partial_unet_state=True, mamba_gate_override=1.0)
ii.run(**kw)
print("[oraclem] per-step usage:", sorted(used.items()), flush=True)
json.dump({"clip": CLIP, "M": MM, "subst": sorted(SUBST), "unet": UNET,
           "usage": {f"{k[0]}_{k[1]}": v for k, v in used.items()}},
          open(os.path.join(SAVE, "bd_oracle_meta.json"), "w"), indent=1)
print("SK_ORACLE_DONE", flush=True)
