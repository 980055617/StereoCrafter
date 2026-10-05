#!/usr/bin/env python
"""deep_20261004 / input_side lane: COPY of scripts/distill/runs/finalcheck_20261004/speed/infer_ll_hook_speed_v1.py
(verbatim copy kept next to this file as infer_ll_hook_speed_v1_ORIG_COPY.py) with ONE addition, inert unless set:

  IS_INPUT_DIR   directory holding warped.npy and mask.npy (uint8 [T,576,1024,3], the deployed 576x1024 window of
                 the model's warped-right-eye and occlusion-mask inputs) and IS_LEFT = the matching left window
                 (uint8 [T,576,1024,3]).  ii.read_and_prepare_video is replaced by a reader that returns these
                 arrays with the SAME arithmetic as utils/inpainting.read_and_prepare_video (uint8 -> torch.tensor ->
                 permute(0,3,1,2) -> float() -> /255.0; mask -> (/255.0).mean(dim=1, keepdim=True)).  The deployed
                 centre crop is then a no-op (the arrays already are the 576x1024 window).  IS_FPS = the splatting
                 video's fps (used only by the writer).  The input md5s are logged.
                 Faithfulness gate: with the DEPLOYED windows (decoded from video_data/splatting/<clip>...mp4) the
                 render must reproduce the existing pre-encode md5 of the same model/clip.

--- original docstring of the copied hook follows ---
finalcheck_20261004 / speed lane: COPY of scripts/distill/runs/skeptic1/infer_ll_hook.py
(the copy as taken is kept next to this file as infer_ll_hook_ORIG_COPY.py).

The deployed-inference path (config, ii.run kwargs, FFV1 writer imported by path from
scripts/distill/runs/fulldata_v2/beyond4/infer_lossless.py, optional 15-tensor swap) is UNCHANGED.
Additions, each INERT unless its env knob is set:

  SK_SIGMAS      comma-separated sigma list ENDING AT sigma_min (e.g. the T5 schedule
                 "700.0,7.276163101196289,1.1675708293914795,0.09738767892122269,0.0020000000949949026").
                 Installed by wrapping EulerDiscreteScheduler.set_timesteps at CLASS level (so main()'s own
                 startup "[sched][inference][timesteps_head]" print shows the custom schedule too).
                 The wrapper drops the pipeline's num_inference_steps and calls the ORIGINAL set_timesteps
                 with sigmas=<list>+[0.0]: the installed diffusers 0.29.2 natively accepts sigmas= but does
                 NOT append the trailing 0 itself (it uses the list as given and sets
                 num_inference_steps = len(sigmas)-1). Natively it then sets
                 timesteps = 0.25*ln(sigma) (timestep_type continuous + v_prediction), num_inference_steps,
                 and resets _step_index/_begin_index -- exactly the EulerDiscreteScheduler code path.
                 Every kept sigma AND its timestep is asserted bit-equal to an entry of the scheduler's own
                 default 8-step schedule computed on the same device.
                 SK_STEPS is ignored when SK_SIGMAS is set (steps = len(list)).
  SK_RNG_PAD_TO  integer M: after the LAST scheduler step of every window, draw (M - N) discarded
                 randn_tensor(model_output.shape, dtype=model_output.dtype, device=model_output.device)
                 samples -- the identical call EulerDiscreteScheduler.step() makes on every step even at
                 gamma=0 (its noise is unused when s_churn=0). An N-step schedule then consumes the global
                 CUDA RNG exactly like the deployed 8-step schedule, so every window starts from the SAME
                 initial noise as the deployed render (paired comparison). A MEASUREMENT DEVICE only:
                 a shipped N-step sampler would run unpadded.
  SK_GUID        min = max guidance (default 1.01, as before; the driver now lets it be inherited).

Passive instrumentation (the c1/c2 md5 controls run WITH it installed, so any perturbation would show):
  per window: CUDA-synchronised wall seconds of the pipeline __call__ (CLIP+VAE encode + sampling loop,
  output_type="latent" so no decode inside), UNet call count + batch sizes (forward pre-hook), UNet GPU ms
  (CUDA events around each forward), md5 of the raw bits of the initial latents returned by
  prepare_latents (the RNG-pairing fingerprint); process: import->first call (model load), total.
  -> <SK_OUT>/speed_log.json and one "[speed]" summary line.
"""
import importlib.util, os, sys, collections, time, json, hashlib

T_IMPORT = time.time()
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
from diffusers.schedulers.scheduling_euler_discrete import EulerDiscreteScheduler as _ES
from diffusers.utils.torch_utils import randn_tensor

CLIP = os.environ["SK_CLIP"]
OUT = os.environ["SK_OUT"]
STEPS = int(os.environ.get("SK_STEPS", "8"))
GUID = float(os.environ.get("SK_GUID", "1.01"))
UNET = os.environ.get("SK_UNET", "").strip() or None
CK = os.environ.get("SK_CK", "").strip()
SIG = [float(x) for x in os.environ.get("SK_SIGMAS", "").split(",") if x.strip()]
PAD_TO = os.environ.get("SK_RNG_PAD_TO", "").strip()
PAD_TO = int(PAD_TO) if PAD_TO else None

assert ii.write_video_opencv is _IL._patched_write, "lossless writer patch did not take"

# ------------------------------------------------------------------ custom sigma schedule (SK_SIGMAS)
_orig_set = _ES.set_timesteps
_orig_step = _ES.step
_REF = {}          # device-string -> (sigmas, timesteps) of the default 8-step schedule
SCHED_LOG = {}

if SIG:
    assert all(SIG[i] > SIG[i + 1] for i in range(len(SIG) - 1)), f"SK_SIGMAS must strictly decrease: {SIG}"
    assert SIG[-1] > 0.0, "SK_SIGMAS must END at sigma_min; the trailing 0 is appended here"
    STEPS = len(SIG)

    def _ref_for(self, device):
        key = str(device)
        if key not in _REF:
            ref = _ES.from_config(self.config)
            _orig_set(ref, 8, device=device)
            _REF[key] = (ref.sigmas.clone(), ref.timesteps.detach().clone().cpu())
        return _REF[key]

    def _set_custom(self, num_inference_steps=None, device=None, timesteps=None, sigmas=None):
        assert timesteps is None and sigmas is None, "custom-sigma wrapper got timesteps/sigmas from caller"
        ref_sig, ref_ts = _ref_for(self, device)
        _orig_set(self, None, device=device, sigmas=list(SIG) + [0.0])
        # ---- checks: exact schedule, exact kept points, reset indices
        assert self.num_inference_steps == len(SIG), self.num_inference_steps
        assert len(self.timesteps) == len(SIG) and len(self.sigmas) == len(SIG) + 1
        assert float(self.sigmas[-1]) == 0.0
        assert self._step_index is None and self._begin_index is None
        ts_cpu = self.timesteps.detach().cpu()
        idx = []
        for k in range(len(SIG)):
            hit = (ref_sig[:-1] == self.sigmas[k]).nonzero().flatten().tolist()
            assert len(hit) == 1, f"sigma {float(self.sigmas[k])!r} is not an exact default-8 sigma"
            j = hit[0]
            assert torch.equal(ts_cpu[k], ref_ts[j]), (k, j, float(ts_cpu[k]), float(ref_ts[j]))
            idx.append(j)
        key = str(device)
        if key not in SCHED_LOG:
            SCHED_LOG[key] = dict(sigmas=[repr(float(s)) for s in self.sigmas],
                                  timesteps=[repr(float(t)) for t in ts_cpu],
                                  default8_index=idx, num_inference_steps=int(self.num_inference_steps))
            print(f"[speed] custom schedule on {key}: sigmas={SCHED_LOG[key]['sigmas']} "
                  f"timesteps={SCHED_LOG[key]['timesteps']} default8_index={idx} "
                  f"(all kept sigma+timestep bit-equal to the default 8-step entries)", flush=True)

    _ES.set_timesteps = _set_custom

# ------------------------------------------------------------------ RNG pad (SK_RNG_PAD_TO)
STATS = dict(pad_draws=0, windows=[])
CUR = {}

if PAD_TO is not None:
    def _step_pad(self, model_output, timestep, sample, *a, **k):
        out = _orig_step(self, model_output, timestep, sample, *a, **k)
        if self._step_index == len(self.timesteps):          # the window's last step just ran
            K = PAD_TO - len(self.timesteps)
            assert K >= 0, f"SK_RNG_PAD_TO={PAD_TO} < steps {len(self.timesteps)}"
            for _ in range(K):
                randn_tensor(model_output.shape, dtype=model_output.dtype, device=model_output.device,
                             generator=k.get("generator"))
            STATS["pad_draws"] += K
            if CUR:
                CUR["pad_draws"] = K
        return out

    _ES.step = _step_pad

print(f"[sk] clip={CLIP} steps={STEPS} guid={GUID} unet={UNET} ck={CK or None} out={OUT} "
      f"sigmas={SIG or None} rng_pad_to={PAD_TO}", flush=True)

SWAP = {}
if CK:
    raw = torch.load(CK, map_location="cpu", weights_only=False)
    SWAP = raw.get("model", raw)
    print(f"[sk] distilled ckpt: {len(SWAP)} tensors", flush=True)

# ------------------------------------------------------------------ passive instrumentation
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
    rec = dict(call_s=t1 - t0,
               unet_calls=len(evs),
               batches=CUR.get("batches", []),
               unet_ms=sum(s.elapsed_time(e) for s, e in evs),
               init_md5=CUR.get("init_md5"),
               init_shape=CUR.get("init_shape"),
               init_dtype=CUR.get("init_dtype"),
               pad_draws=CUR.get("pad_draws", 0),
               num_frames=k.get("num_frames"))
    STATS["windows"].append(rec)
    return out


ii._Pipe.__call__ = wrapped_call

# ------------------------------------------------------------------ input_side: array reader (IS_INPUT_DIR)
IS_INPUT_DIR = os.environ.get("IS_INPUT_DIR", "").strip()
if IS_INPUT_DIR:
    import numpy as _np
    _IS_LEFT = os.environ["IS_LEFT"]
    _IS_FPS = float(os.environ["IS_FPS"])

    def _md5_file(p):
        h = hashlib.md5()
        with open(p, "rb") as fh:
            for blk in iter(lambda: fh.read(1 << 24), b""):
                h.update(blk)
        return h.hexdigest()

    _paths = dict(left=_IS_LEFT, warped=os.path.join(IS_INPUT_DIR, "warped.npy"),
                  mask=os.path.join(IS_INPUT_DIR, "mask.npy"))
    _md5s = {k: _md5_file(v) for k, v in _paths.items()}
    print(f"[is] input arrays: {_paths} md5 {_md5s}", flush=True)

    def _is_reader(input_video_path, return_right=False):
        assert not return_right
        assert os.path.basename(input_video_path) == f"{CLIP}_splatting_results.mp4", input_video_path
        L = _np.load(_paths["left"])
        Wp = _np.load(_paths["warped"])
        M = _np.load(_paths["mask"])
        assert L.dtype == Wp.dtype == M.dtype == _np.uint8
        assert L.shape == Wp.shape == M.shape and L.shape[1:] == (576, 1024, 3), (L.shape, Wp.shape, M.shape)
        frames_left = torch.tensor(L).permute(0, 3, 1, 2).float() / 255.0
        frames_warped = torch.tensor(Wp).permute(0, 3, 1, 2).float() / 255.0
        frames_mask = (torch.tensor(M).permute(0, 3, 1, 2).float() / 255.0).mean(dim=1, keepdim=True)
        print(f"[is] reader returned T={len(Wp)} fps={_IS_FPS} (arrays from {IS_INPUT_DIR})", flush=True)
        return _IS_FPS, frames_left, frames_warped, frames_mask

    ii.read_and_prepare_video = _is_reader

kw = dict(config="config/0160_overfit_inference_matched.json",
          input_video_path=f"video_data/splatting/{CLIP}_splatting_results.mp4",
          save_dir=OUT, num_inference_steps=STEPS,
          min_guidance_scale=GUID, max_guidance_scale=GUID)
if UNET is None:
    kw["unet_state_path"] = None
else:
    kw["unet_state_path"] = UNET
    kw["expected_partial_unet_state"] = True
    kw["mamba_gate_override"] = 1.0
ii.run(**kw)
T_END = time.time()

W = STATS["windows"]
summary = dict(clip=CLIP, out=OUT, guid=GUID, steps=STEPS, sigmas=SIG or None, rng_pad_to=PAD_TO,
               is_input_dir=IS_INPUT_DIR or None, is_input_md5=(_md5s if IS_INPUT_DIR else None),
               unet_state=UNET, ck=CK or None,
               n_windows=len(W),
               unet_calls_total=sum(w["unet_calls"] for w in W),
               unet_calls_per_window=sorted(set(w["unet_calls"] for w in W)),
               batch_sizes=sorted(set(b for w in W for b in w["batches"])),
               call_s_sum=sum(w["call_s"] for w in W),
               unet_ms_sum=sum(w["unet_ms"] for w in W),
               pad_draws_total=STATS["pad_draws"],
               load_s=(T_FIRST_CALL - T_IMPORT) if T_FIRST_CALL else None,
               hook_total_s=T_END - T_IMPORT,
               schedule=SCHED_LOG or None,
               init_md5=[w["init_md5"] for w in W],
               windows=W)
with open(os.path.join(OUT, "speed_log.json"), "w") as fh:
    json.dump(summary, fh, indent=1)
print(f"[speed] windows={summary['n_windows']} unet_calls={summary['unet_calls_total']} "
      f"per_window={summary['unet_calls_per_window']} batch={summary['batch_sizes']} "
      f"call_s_sum={summary['call_s_sum']:.2f} unet_s_sum={summary['unet_ms_sum'] / 1000:.2f} "
      f"load_s={summary['load_s']:.1f} hook_total_s={summary['hook_total_s']:.1f} "
      f"pad_draws={summary['pad_draws_total']} "
      f"init_md5_all={hashlib.md5(''.join(summary['init_md5']).encode()).hexdigest()}", flush=True)
print("[sk] done", flush=True)
