#!/usr/bin/env python
"""more_20261004 / teacher lane: COPY of scripts/distill/runs/finalcheck_20261004/speed/infer_ll_hook_speed_v1.py
with its SK_SIGMAS / SK_RNG_PAD_TO machinery REMOVED and one knob set ADDED.  The deployed-inference path (config,
ii.run kwargs, FFV1 writer imported by path from scripts/distill/runs/fulldata_v2/beyond4/infer_lossless.py) and
the passive instrumentation (per-window CUDA-synchronised call time, UNet calls/batches/GPU ms, md5 of the initial
latents = RNG-pairing fingerprint) are UNCHANGED.

Added (inert unless SK_SAMPLER is set):
  SK_SAMPLER   euler | heun | dpmpp2m | euler_churn  -> on the first pipeline __call__ the pipeline's
               EulerDiscreteScheduler is REPLACED by teacher_sched_v1.TeacherScheduler built from its own config
               (object.__setattr__, bypassing DiffusionPipeline's config registration).  Empty = no swap.
  SK_N         number of Karras sigmas (steps) for the sampler; also passed as num_inference_steps.
  SK_PAD       RNG pad_to (draws per window) -- 25 pairs every window's initial noise with the s25 render.
  SK_SCHURN SK_STMIN SK_STMAX SK_SNOISE   EDM churn parameters (euler_churn only).
  SK_MAXCHUNKS max_profile_chunks for 2-window smokes (output video truncated; never scored).
  A UNet forward pre-hook asserts on EVERY evaluation that the `t` the UNet receives is the scheduler's
  timesteps[k] AND equals 0.25*ln(sigma_eval[k]) to fp32 precision; sigma_eval of window 0 and of the last
  window are logged in full.
"""
import importlib.util, os, sys, time, json, hashlib, math

T_IMPORT = time.time()
REPO = "/home/kawa/master_project/StereoCrafter"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)
os.chdir(REPO)

_spec = importlib.util.spec_from_file_location(
    "infer_lossless", f"{REPO}/scripts/distill/runs/fulldata_v2/beyond4/infer_lossless.py")
_IL = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_IL)

import torch
import inpainting_inference as ii
from teacher_sched_v1 import TeacherScheduler

CLIP = os.environ["SK_CLIP"]
OUT = os.environ["SK_OUT"]
GUID = float(os.environ.get("SK_GUID", "1.01"))
UNET = os.environ.get("SK_UNET", "").strip() or None
SAMPLER = os.environ.get("SK_SAMPLER", "").strip() or None
N = int(os.environ.get("SK_N", "8"))
PAD = os.environ.get("SK_PAD", "").strip()
PAD = int(PAD) if PAD else None
CHURN = dict(s_churn=float(os.environ.get("SK_SCHURN", "0") or 0),
             s_tmin=float(os.environ.get("SK_STMIN", "0") or 0),
             s_tmax=float(os.environ.get("SK_STMAX", "inf") or "inf"),
             s_noise=float(os.environ.get("SK_SNOISE", "1") or 1))
MAXCH = os.environ.get("SK_MAXCHUNKS", "").strip()
MAXCH = int(MAXCH) if MAXCH else None

assert ii.write_video_opencv is _IL._patched_write, "lossless writer patch did not take"
print(f"[sk] clip={CLIP} sampler={SAMPLER} N={N} pad={PAD} churn={CHURN if SAMPLER == 'euler_churn' else None} "
      f"guid={GUID} unet={UNET} out={OUT} maxchunks={MAXCH}", flush=True)

STATS = dict(windows=[])
CUR = {}
SCHED = {}

_orig_prep = ii._Pipe.prepare_latents


def _prep_fp(self, *a, **k):
    lat = _orig_prep(self, *a, **k)
    with torch.no_grad():
        raw_bits = lat.detach().contiguous().view(torch.int16) if lat.element_size() == 2 else lat.detach().contiguous()
        CUR["init_md5"] = hashlib.md5(raw_bits.cpu().numpy().tobytes()).hexdigest()
        CUR["init_shape"] = list(lat.shape)
        CUR["init_dtype"] = str(lat.dtype)
    return lat


ii._Pipe.prepare_latents = _prep_fp


def _unet_pre(mod, args, kwargs):
    sample = args[0] if args else kwargs["sample"]
    CUR.setdefault("batches", []).append(int(sample.shape[0]))
    sch = SCHED.get("s")
    if sch is not None:
        t = args[1] if len(args) > 1 else kwargs["timestep"]
        k = sch._k
        assert torch.equal(torch.as_tensor(t).to(sch.timesteps.device), sch.timesteps[k]), \
            f"UNet got t={t} but scheduler timesteps[{k}]={sch.timesteps[k]}"
        sg = float(sch.eval_sigmas[k])
        assert abs(float(t) - 0.25 * math.log(sg)) <= 2e-7 * max(1.0, abs(float(t))), (float(t), sg)
        CUR.setdefault("sigma_eval", []).append(sg)
    ev = torch.cuda.Event(enable_timing=True)
    ev.record()
    CUR.setdefault("ev", []).append([ev, None])


def _unet_post(mod, args, kwargs, output):
    ev = torch.cuda.Event(enable_timing=True)
    ev.record()
    CUR["ev"][-1][1] = ev


T_FIRST_CALL = None
_call = ii._Pipe.__call__


def wrapped_call(self, *a, **k):
    global T_FIRST_CALL
    if T_FIRST_CALL is None:
        T_FIRST_CALL = time.time()
    if SAMPLER and "s" not in SCHED:
        old = self.scheduler
        sch = TeacherScheduler(old.config, SAMPLER, N, pad_to=PAD, **(CHURN if SAMPLER == "euler_churn" else {}))
        object.__setattr__(self, "scheduler", sch)
        assert self.scheduler is sch
        SCHED["s"] = sch
        print(f"[sk] scheduler swapped: {type(old).__name__} -> TeacherScheduler(mode={SAMPLER}, N={N}, pad_to={PAD})",
              flush=True)
    unet = self.unet
    if not getattr(unet, "_speed_hooks", False):
        unet.register_forward_pre_hook(_unet_pre, with_kwargs=True)
        unet.register_forward_hook(_unet_post, with_kwargs=True)
        unet._speed_hooks = True
    CUR.clear()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = _call(self, *a, **k)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    evs = CUR.pop("ev", [])
    sch = SCHED.get("s")
    wl = sch.window_log[-1] if sch is not None and sch.window_log else None
    rec = dict(call_s=t1 - t0,
               unet_calls=len(evs),
               batches=CUR.get("batches", []),
               unet_ms=sum(s.elapsed_time(e) for s, e in evs),
               init_md5=CUR.get("init_md5"),
               init_shape=CUR.get("init_shape"),
               init_dtype=CUR.get("init_dtype"),
               num_frames=k.get("num_frames"),
               sched_evals=None if wl is None else wl["evals"],
               sched_draws=None if wl is None else wl["draws"],
               sched_pad=None if wl is None else wl["pad"],
               sigma_eval=CUR.get("sigma_eval"))
    if sch is not None:
        assert rec["unet_calls"] == len(sch.timesteps) == wl["evals"], (rec["unet_calls"], len(sch.timesteps), wl)
        if PAD is not None:
            assert wl["draws"] + wl["pad"] == PAD, wl
    STATS["windows"].append(rec)
    return out


ii._Pipe.__call__ = wrapped_call

kw = dict(config="config/0160_overfit_inference_matched.json",
          input_video_path=f"video_data/splatting/{CLIP}_splatting_results.mp4",
          save_dir=OUT, num_inference_steps=N,
          min_guidance_scale=GUID, max_guidance_scale=GUID)
if UNET is None:
    kw["unet_state_path"] = None
else:
    kw["unet_state_path"] = UNET
    kw["expected_partial_unet_state"] = True
    kw["mamba_gate_override"] = 1.0
if MAXCH is not None:
    kw["max_profile_chunks"] = MAXCH
ii.run(**kw)
T_END = time.time()

W = STATS["windows"]
for i, w in enumerate(W):          # keep the full sigma_eval list only for window 0 and the last window
    if 0 < i < len(W) - 1:
        w["sigma_eval"] = None if w["sigma_eval"] is None else len(w["sigma_eval"])
sch = SCHED.get("s")
summary = dict(clip=CLIP, out=OUT, guid=GUID, sampler=SAMPLER, N=N, pad_to=PAD,
               churn=CHURN if SAMPLER == "euler_churn" else None,
               gammas=None if sch is None else sch.gammas,
               karras_sigmas=None if sch is None else [repr(float(s)) for s in sch.sigmas],
               eval_sigmas=None if sch is None else [repr(float(s)) for s in sch.eval_sigmas],
               timesteps=None if sch is None else [repr(float(t)) for t in sch.timesteps.cpu()],
               unet_state=UNET, maxchunks=MAXCH,
               n_windows=len(W),
               unet_calls_total=sum(w["unet_calls"] for w in W),
               unet_calls_per_window=sorted(set(w["unet_calls"] for w in W)),
               batch_sizes=sorted(set(b for w in W for b in w["batches"])),
               draws_per_window=sorted(set((w["sched_draws"] or 0) + (w["sched_pad"] or 0) for w in W)),
               call_s_sum=sum(w["call_s"] for w in W),
               unet_ms_sum=sum(w["unet_ms"] for w in W),
               load_s=(T_FIRST_CALL - T_IMPORT) if T_FIRST_CALL else None,
               hook_total_s=T_END - T_IMPORT,
               init_md5=[w["init_md5"] for w in W],
               windows=W)
with open(os.path.join(OUT, "speed_log.json"), "w") as fh:
    json.dump(summary, fh, indent=1)
print(f"[speed] windows={summary['n_windows']} unet_calls={summary['unet_calls_total']} "
      f"per_window={summary['unet_calls_per_window']} batch={summary['batch_sizes']} "
      f"draws_per_window={summary['draws_per_window']} "
      f"call_s_sum={summary['call_s_sum']:.2f} unet_s_sum={summary['unet_ms_sum'] / 1000:.2f} "
      f"load_s={summary['load_s']:.1f} hook_total_s={summary['hook_total_s']:.1f} "
      f"init_md5_all={hashlib.md5(''.join(summary['init_md5']).encode()).hexdigest()}", flush=True)
print("[sk] done", flush=True)
