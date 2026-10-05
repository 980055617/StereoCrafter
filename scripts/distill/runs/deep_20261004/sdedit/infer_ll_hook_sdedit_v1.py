#!/usr/bin/env python
"""deep_20261004 / sdedit lane, hook v1 = COPY of scripts/distill/runs/finalcheck_20261004/speed/infer_ll_hook_speed_v1.py
(kept verbatim next to this file as infer_ll_hook_speed_v1_ORIG_COPY.py, md5 7f5bf0becdd96e43a4a09075f35ad29d) plus ONE
addition, INERT unless SD_MODE != off (the K1 control proves it: SD_MODE=off must reproduce the speed lane's writer md5).

  SD_MODE   off (default) | warp | warpfill
            SDEdit-style start.  The sampler starts at sigma_start = the FIRST entry of SK_SIGMAS (which must be a tail
            of the deployed default-8 Karras grid; the speed hook's own set_timesteps wrapper asserts every kept sigma
            and timestep bit-equal to the default-8 entries) from
                x_init = sf * z_src + sigma_start * eps            (EDM forward process x_sigma = x0 + sigma * eps,
                                                                    which is what EulerDiscreteScheduler's v-pred
                                                                    c_skip / c_out / c_in assume)
            instead of the pipeline's  eps * init_noise_sigma,  where
              sf    = pipeline.vae.config.scaling_factor (0.18215; decode_latents divides by it)
              eps   = randn_tensor(shape, generator=generator, device=device, dtype=dtype): the IDENTICAL call that
                      MambaStableVideoDiffusionInpaintingPipeline.prepare_latents makes, so the CUDA RNG is consumed
                      exactly as in the deployed render.  With SK_RNG_PAD_TO=8 (required) every window's eps is the
                      deployed 8-step render's own initial noise of that window (paired comparison).
              z_src = warp      the window's CONDITIONING latents themselves: the pipeline's own _encode_vae_frames
                                output (VAE latent_dist.mode() of the window's warped right-eye frames), captured.
                      warpfill  the same encode (image_processor.preprocess + _encode_vae_frames, chunks of 5) of the
                                window's warped frames with EVERY hole pixel (mask >= 0.5, the pipeline's own
                                mask_processor binarisation) filled by row-wise linear interpolation between the nearest
                                non-hole pixels left and right of the run (crackfill_COPY.fill_rowlin applied to ALL hole
                                runs, any length; a run that spans a whole row would be left untouched and counted).
                                Computed per frame on the deployed centre-crop window + SD_MARGIN (16) px, then cropped.
                                The CONDITIONING latents, the CLIP image embedding and the mask channel are NOT changed:
                                the filled frames reach z_src only (looked up by md5 of the unfilled window frame).
            Requirements (asserted): SK_SIGMAS set with sigma_start < 700, SK_RNG_PAD_TO=8, guidance 1.00 (no CFG).
            Writes <SK_OUT>/sdedit_log.json, per window:
              paired_ref_md5  md5 of the raw bits of  eps * (default-8 init_noise_sigma)  in the latents dtype = exactly
                              what the deployed prepare_latents would have returned from this draw (K2 check against
                              <clip>_*_g100_s8/speed_log.json init_md5 of the speed lane)
              statistics of sf*z_src, of x_init and of (x_init - sf*z_src)/sigma_start; for warpfill the mean
              |sf*(z_fill - z_cond)| (the fill's latent footprint, to set against sigma_start) and fill pixel counts.
            Note: the speed hook's per-window init_md5 (speed_log.json) is the md5 of x_init here, not of the noise.
------------------------------------------------------------------------------------------------------------------------
ORIGINAL DOCSTRING (speed lane):
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

# ================================================================== sdedit lane addition (SD_*), inert when SD_MODE=off
import numpy as _np  # noqa: E402

SD_MODE = os.environ.get("SD_MODE", "off").strip() or "off"
SD_MARGIN = int(os.environ.get("SD_MARGIN", "16"))
assert SD_MODE in ("off", "warp", "warpfill"), SD_MODE
SDC = {}                       # per-window state, set by wrapped_call / the encode capture
SD_LOG = dict(mode=SD_MODE, margin=SD_MARGIN, windows=[], fill=None)
FILL_MAP = {}
_orig_encode_vae_frames = ii._Pipe._encode_vae_frames
_REF_INS = {}


def _md5_bits(t):
    t = t.detach().contiguous()
    raw_bits = t.view(torch.int16) if t.element_size() == 2 else t
    return hashlib.md5(raw_bits.cpu().numpy().tobytes()).hexdigest()


def _ref_init_noise_sigma(self, device):
    """init_noise_sigma of the scheduler's own DEFAULT 8-step schedule (what the deployed prepare_latents multiplies by)."""
    key = str(device)
    if key not in _REF_INS:
        ref = _ES.from_config(self.scheduler.config)
        _orig_set(ref, 8, device=device)
        _REF_INS[key] = ref.init_noise_sigma
    return _REF_INS[key]


if SD_MODE != "off":
    assert SIG, "SD_MODE needs SK_SIGMAS = the tail of the deployed default-8 grid that starts at sigma_start"
    assert SIG[0] < 700.0, f"sigma_start must be below sigma_max, got {SIG[0]}"
    assert PAD_TO == 8, "SD_MODE requires SK_RNG_PAD_TO=8 (eps = the deployed window's own initial noise)"
    assert GUID == 1.0, "SD_MODE is defined at guidance 1.00 (no CFG batch)"
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import sdfill_v1 as _SDF  # noqa: E402  (row-wise linear fill of every hole run; uses crackfill_COPY.fill_rowlin)

    def _enc_capture(self, frames, *a, **k):
        out = _orig_encode_vae_frames(self, frames, *a, **k)
        SDC["cond"] = out
        SDC["cond_calls"] = SDC.get("cond_calls", 0) + 1
        return out

    ii._Pipe._encode_vae_frames = _enc_capture

    def _sd_prepare(self, batch_size, num_frames, num_channels_latents, height, width, dtype, device, generator,
                    latents=None):
        assert latents is None, "SDEdit init expects the pipeline to draw its own noise"
        assert not self.do_classifier_free_guidance, "SDEdit init is defined without the CFG batch"
        shape = (batch_size, num_frames, num_channels_latents // 2,
                 height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError("generator list / batch size mismatch")
        eps = randn_tensor(shape, generator=generator, device=device, dtype=dtype)   # == prepare_latents' own draw
        paired_ref_md5 = _md5_bits(eps * _ref_init_noise_sigma(self, device))
        s0 = float(self.scheduler.sigmas[0])
        assert s0 == SIG[0], (s0, SIG[0])                       # the scheduler starts at sigma_start (bit-equal)
        sf = float(self.vae.config.scaling_factor)
        assert SDC.get("cond_calls") == 1, f"expected one cond encode before prepare_latents, got {SDC.get('cond_calls')}"
        zc = SDC["cond"]
        assert zc.shape[0] == 1 and tuple(zc.shape[1:]) == tuple(shape[1:]), (tuple(zc.shape), shape)
        rec = dict(sigma_start=repr(s0), sf=sf, paired_ref_md5=paired_ref_md5, shape=list(shape), dtype=str(dtype))
        if SD_MODE == "warp":
            z = zc
        else:
            fr = SDC["filled"]
            assert fr.shape[0] == num_frames, (fr.shape, num_frames)
            ff = self.image_processor.preprocess(fr, height=height, width=width)
            z = _orig_encode_vae_frames(self, ff, device, 1, False, n_frames_per_time=5)
            assert tuple(z.shape) == tuple(zc.shape), (tuple(z.shape), tuple(zc.shape))
            dz = (z.float() - zc.float()) * sf
            rec["fill_latent_mean_abs_diff"] = float(dz.abs().mean())
            rec["fill_latent_max_abs_diff"] = float(dz.abs().max())
            rec["fill_latent_frac_changed"] = float((dz != 0).float().mean())
            rec["filled_frames_md5"] = hashlib.md5(fr.contiguous().numpy().tobytes()).hexdigest()
        x0 = z.to(device=device, dtype=torch.float32) * sf
        x = x0 + s0 * eps.float()
        lat = x.to(dtype)
        r = (lat.float() - x0) / s0
        rec.update(src_mean=float(x0.mean()), src_std=float(x0.std()), init_mean=float(lat.float().mean()),
                   init_std=float(lat.float().std()), resid_over_sigma_std=float(r.std()),
                   resid_over_sigma_mean=float(r.mean()), eps_std=float(eps.float().std()))
        SD_LOG["windows"].append(rec)
        return lat

    if SD_MODE == "warpfill":
        _CFG = json.load(open(f"{REPO}/config/0160_overfit_inference_matched.json"))
        _TH, _TW = int(_CFG["target_height"]), int(_CFG["target_width"])
        assert float(_CFG.get("overlap_prev_weight", 1.0)) <= 0.0, "fill lookup assumes the overlap frames stay the warped input"
        _orig_read = ii.read_and_prepare_video
        _fill_all_rowlin = _SDF.fill_all_rowlin

        def _read_fillmap(path, *a, **k):
            out = _orig_read(path, *a, **k)
            fps, fl, fw, fm = out[:4]
            T, C, H, W = fw.shape
            top, left = (H - _TH) // 2, (W - _TW) // 2           # == inpainting_inference._center_crop_frames
            y0, y1 = max(0, top - SD_MARGIN), min(H, top + _TH + SD_MARGIN)
            x0_, x1_ = max(0, left - SD_MARGIN), min(W, left + _TW + SD_MARGIN)
            wy0, wx0 = top - y0, left - x0_
            t_f = time.time()
            hist = collections.Counter()
            n_hole_win = n_chg_win = n_full = n_runs = 0
            per_frame = []
            md5_unf = hashlib.md5()
            for t in range(T):
                reg = fw[t, :, y0:y1, x0_:x1_].permute(1, 2, 0).contiguous().numpy()
                holes = fm[t, 0, y0:y1, x0_:x1_].numpy() >= 0.5
                filled, st = _fill_all_rowlin(reg, holes)
                win_unf = fw[t, :, top:top + _TH, left:left + _TW].contiguous()
                key = hashlib.md5(win_unf.numpy().tobytes()).hexdigest()
                md5_unf.update(key.encode())
                win_fill = torch.from_numpy(_np.ascontiguousarray(filled[wy0:wy0 + _TH, wx0:wx0 + _TW])).permute(2, 0, 1).contiguous()
                if key in FILL_MAP:
                    assert torch.equal(FILL_MAP[key], win_fill), "two identical warped frames got different fills"
                FILL_MAP[key] = win_fill
                hw = holes[wy0:wy0 + _TH, wx0:wx0 + _TW]
                n_hole_win += int(hw.sum())
                n_chg_win += int(st["changed"][wy0:wy0 + _TH, wx0:wx0 + _TW].sum())
                n_full += st["full_rows"]
                n_runs += st["runs"]
                for L_, c_ in zip(*_np.unique(_np.minimum(st["lens"], 9), return_counts=True)):
                    hist[int(L_)] += int(c_)
                per_frame.append(float(hw.mean()))
            # the tensors handed back are UNCHANGED (the conditioning path never sees the fill)
            SD_LOG["fill"] = dict(input_path=path, frames=int(T), window=[top, top + _TH, left, left + _TW],
                                  region=[y0, y1, x0_, x1_], hole_px_window=n_hole_win, changed_px_window=n_chg_win,
                                  hole_frac_window=n_hole_win / float(T * _TH * _TW), runs_region=n_runs,
                                  full_row_runs_left_unfilled=n_full,
                                  runlen_hist_region_capped9={str(k_): v_ for k_, v_ in sorted(hist.items())},
                                  hole_frac_per_frame=per_frame, map_entries=len(FILL_MAP),
                                  md5_of_unfilled_window_md5s=md5_unf.hexdigest(), fill_seconds=time.time() - t_f)
            print(f"[sdedit] fill map: frames={T} entries={len(FILL_MAP)} hole_frac_window={SD_LOG['fill']['hole_frac_window']:.5f} "
                  f"changed_px_window={n_chg_win}/{n_hole_win} runs={n_runs} full_rows={n_full} "
                  f"runlen_hist={SD_LOG['fill']['runlen_hist_region_capped9']} s={SD_LOG['fill']['fill_seconds']:.1f}",
                  flush=True)
            return out

        ii.read_and_prepare_video = _read_fillmap

print(f"[sdedit] mode={SD_MODE} sigma_start={SIG[0] if (SIG and SD_MODE != 'off') else None} margin={SD_MARGIN}",
      flush=True)

# ------------------------------------------------------------------ passive instrumentation
_orig_prep = ii._Pipe.prepare_latents if SD_MODE == "off" else _sd_prepare


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
    if SD_MODE != "off":                                   # sdedit lane: per-window state
        SDC.clear()
        if SD_MODE == "warpfill":
            fr = k["frames"]
            keys = [hashlib.md5(fr[i].contiguous().numpy().tobytes()).hexdigest() for i in range(fr.shape[0])]
            miss = [i for i, kk in enumerate(keys) if kk not in FILL_MAP]
            assert not miss, f"warpfill: {len(miss)} conditioning frames not found in the fill map (frames {miss[:5]})"
            SDC["filled"] = torch.stack([FILL_MAP[kk] for kk in keys])
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
if SD_MODE != "off":
    assert len(SD_LOG["windows"]) == len(W), (len(SD_LOG["windows"]), len(W))
    SD_LOG["paired_ref_md5"] = [w["paired_ref_md5"] for w in SD_LOG["windows"]]
    SD_LOG["paired_ref_md5_all"] = hashlib.md5("".join(SD_LOG["paired_ref_md5"]).encode()).hexdigest()
    with open(os.path.join(OUT, "sdedit_log.json"), "w") as fh:
        json.dump(SD_LOG, fh, indent=1)
    ws = SD_LOG["windows"]
    extra = ""
    if SD_MODE == "warpfill":
        extra = (f" fill_lat_mean|d|={_np.mean([w['fill_latent_mean_abs_diff'] for w in ws]):.5f}"
                 f" fill_lat_max|d|={max(w['fill_latent_max_abs_diff'] for w in ws):.4f}")
    print(f"[sdedit] windows={len(ws)} sigma_start={ws[0]['sigma_start']} "
          f"paired_ref_md5_all={SD_LOG['paired_ref_md5_all']} src_std={_np.mean([w['src_std'] for w in ws]):.4f} "
          f"init_std={_np.mean([w['init_std'] for w in ws]):.4f} "
          f"resid/sigma_std={_np.mean([w['resid_over_sigma_std'] for w in ws]):.4f}{extra}", flush=True)
print("[sk] done", flush=True)
