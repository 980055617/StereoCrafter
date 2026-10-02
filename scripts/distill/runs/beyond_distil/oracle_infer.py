"""ORACLE inference: the deployed 8-step pipeline with each Euler step REPLACED by the fine sub-integration.

This is the discriminating sanity check the task demands: if the target construction is right, substituting
x0hat_target at every k must produce an output whose LPIPS is close to the s25 number for that clip
(0301 0.4083, 0204 0.1950) and NOT the deployed number (0.4445 / 0.2122).  Nothing is trained here.

The substitution is done on the LATENT (prev_sample := x_target) rather than on v, because at k=7
(sigma 0.002 -> 0) recovering v from x0hat divides by c_out = 0.002 and is numerically hopeless; the latent
form is exact and equivalent for every k.  scheduler.step is still called first, so the step index and the
per-step randn_tensor RNG draw stay bit-identical to deployment.

Writes a normal *_inpainting_results_sbs.mp4 through inpainting_inference.run, so scripts/distill/score_clip.py
scores it exactly like every other row.

usage: CUDA_VISIBLE_DEVICES=1 BD_M=4 BD_SUBST=all python oracle_infer.py <clip> <save_dir>
env    BD_M      substeps per coarse interval (default 4; 24/7=3.43 matches the 25-step Karras density)
       BD_SUBST  "all" or a comma list of coarse step indices, e.g. "6" or "5,6"
"""
import os, sys, math, json, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bdlib as B
import torch
import inpainting_inference as ii

CLIP, SAVE = sys.argv[1], sys.argv[2]
MM = int(os.environ.get("BD_M", "4"))
SS = os.environ.get("BD_SUBST", "all")
SUBST = set(range(8)) if SS == "all" else {int(x) for x in SS.split(",")}
print(f"[oracle] clip {CLIP} substeps M={MM} substituted coarse steps {sorted(SUBST)}", flush=True)

stash = {}
used = collections.Counter()
_call = ii._Pipe.__call__


def wrapped_call(self, *a, **k):
    if not getattr(self.unet, "_bd_installed", False):
        unet = self.unet
        sch = self.scheduler
        _fwd = unet.forward

        def fwd(sample, timestep, *fa, **fk):
            stash["frame"] = sample[:, :, 4:8]
            stash["mask"] = sample[:, :, 8:9]
            stash["emb"] = fk.get("encoder_hidden_states", fa[0] if fa else None)
            stash["add"] = fk.get("added_time_ids", None)
            return _fwd(sample, timestep, *fa, **fk)

        _step = sch.step

        def step(model_output, timestep, sample, **kw):
            kk = sch.step_index if sch.step_index is not None else 0
            sg = float(sch.sigmas[kk])
            sg1 = float(sch.sigmas[kk + 1])
            out = _step(model_output, timestep, sample, **kw)       # advances step_index, consumes the RNG draw
            if kk not in SUBST:
                used[(kk, "coarse")] += 1
                return out
            xt = B.fine_step(unet, sample.float(), sg, sg1, MM, stash["frame"], stash["mask"],
                             stash["emb"], stash["add"], guid=B.GUID, v0=model_output.float(),
                             dt=model_output.dtype)
            used[(kk, "fine")] += 1
            out.prev_sample = xt.to(model_output.dtype)
            return out

        unet.forward = fwd
        sch.step = step
        unet._bd_installed = True
    return _call(self, *a, **k)


ii._Pipe.__call__ = wrapped_call
ii.run(config="config/0160_overfit_inference_matched.json", unet_state_path=None,
       input_video_path=f"video_data/splatting/{CLIP}_splatting_results.mp4", save_dir=SAVE)
print("[oracle] per-step usage:", sorted(used.items()), flush=True)
json.dump({"clip": CLIP, "M": MM, "subst": sorted(SUBST),
           "usage": {f"{k[0]}_{k[1]}": v for k, v in used.items()}},
          open(os.path.join(SAVE, "bd_oracle_meta.json"), "w"), indent=1)
print("BD_ORACLE_DONE", flush=True)
