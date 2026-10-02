"""ZERO-COST control: the deployed 8-step sampler with each Euler step's LENGTH rescaled by a per-step scalar.

Section F of target_check.py fits  x0hat_target - x0hat_k ~ alpha_k * (x0hat_k - x_k).  If that fit is good, the
"follow the fine trajectory" correction is not a new function at all, it is a step-size correction, and

    x0hat_target = (1+alpha)*x0hat - alpha*x_k
    x_{k+1}      = r*x_k + (1-r)*x0hat_target = x_k + (1+alpha)*(x_{k+1}^Euler - x_k)

i.e.  prev_sample := x_k + beta_k*(prev_sample^Euler - x_k),  beta_k = 1 + alpha_k.  ZERO extra UNet calls, so
this runs at EXACTLY deployed cost -- strictly cheaper than any trained student.  It is the control that says
whether the trained 15 tensors are buying anything a one-line scheduler change would not.

usage: CUDA_VISIBLE_DEVICES=1 BD_BETA="4:1.05,5:1.12,6:1.02" python rescale_infer.py <clip> <save_dir>
       (steps not listed keep beta = 1, i.e. the deployed step)
"""
import os, sys, json, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bdlib as B  # noqa: F401  (chdirs to REPO, disables Mamba, same import contract as oracle_infer)
import torch
import inpainting_inference as ii

CLIP, SAVE = sys.argv[1], sys.argv[2]
BETA = {int(a.split(":")[0]): float(a.split(":")[1]) for a in os.environ["BD_BETA"].split(",")}
print(f"[rescale] clip {CLIP} beta {BETA}", flush=True)
used = collections.Counter()
_call = ii._Pipe.__call__


def wrapped_call(self, *a, **k):
    if not getattr(self.scheduler, "_bd_rescale", False):
        sch = self.scheduler
        _step = sch.step

        def step(model_output, timestep, sample, **kw):
            kk = sch.step_index if sch.step_index is not None else 0
            out = _step(model_output, timestep, sample, **kw)
            b = BETA.get(kk, 1.0)
            used[(kk, b)] += 1
            if b != 1.0:
                s = sample.float()
                out.prev_sample = (s + b * (out.prev_sample.float() - s)).to(model_output.dtype)
            return out

        sch.step = step
        sch._bd_rescale = True
    return _call(self, *a, **k)


ii._Pipe.__call__ = wrapped_call
ii.run(config="config/0160_overfit_inference_matched.json", unet_state_path=None,
       input_video_path=f"video_data/splatting/{CLIP}_splatting_results.mp4", save_dir=SAVE)
print("[rescale] per-step usage:", sorted(used.items()), flush=True)
json.dump({"clip": CLIP, "beta": {str(a): b for a, b in BETA.items()},
           "usage": {f"k{a}_b{b}": v for (a, b), v in used.items()}},
          open(os.path.join(SAVE, "bd_rescale_meta.json"), "w"), indent=1)
print("BD_RESCALE_DONE", flush=True)
