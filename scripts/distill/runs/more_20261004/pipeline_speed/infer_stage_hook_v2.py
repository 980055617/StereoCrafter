#!/usr/bin/env python
"""[v2 = v1 + passive timers around VaeImageProcessor.preprocess (img_preprocess / mask_preprocess).]
more_20261004 / pipeline_speed: STAGE-INSTRUMENTED copy of
scripts/distill/runs/finalcheck_20261004/independent/infer_ll_hook_speed_res_v1.py (md5 13c805716d925148d931f0c4dfe0c854,
kept next to this file as infer_ll_hook_speed_res_v1_ORIG_COPY.py).

UNCHANGED from the copy: the deployed inference call (config/0160_overfit_inference_matched.json, ii.run kwargs),
the FFV1 writer imported by path from scripts/distill/runs/fulldata_v2/beyond4/infer_lossless.py, SK_SIGMAS,
SK_RNG_PAD_TO, SK_H/SK_W (+ tile_num=1), the passive UNet CUDA-event timing and init-latent fingerprints.
CHANGED:
  * reader: the TRACKED utils.inpainting reader by default (what the 576 speed lane and the deployed CLI use);
    SK_READER=lowmem rebinds the copied lowmem reader (tensor-identical, proven by validate/independent);
    SK_READER=cropfirst rebinds reader_cropfirst.read_cropfirst (tensor-identical, reader_bench_v1.py).
  * passive STAGE timers: every wrapped call is bracketed by torch.cuda.synchronize() + perf_counter and appended to
    a timeline -> <SK_OUT>/stage_log.json (+ one "[stage]" summary line).
  * fix knobs, each INERT unless its env var is set:
      SK_DECODE_CHUNK=N      decode_chunk_size passed to main()            (deployed 2)
      SK_CUDNN_BENCH=1       torch.backends.cudnn.benchmark = True
      SK_CHANNELS_LAST=vae,unet   Conv2d weights -> channels_last before the first pipeline call (Conv3d kept)
      SK_SKIP_DISCARDED=1    decode_latents skips every decode chunk whose frames main() discards anyway
                             (frames [0, cur_overlap) of windows i>0); kept frames are decoded from the identical
                             chunk inputs, discarded positions are filled with zeros
      SK_NOAUG_SKIP=1        the CPU randn for noise_aug is replaced by a 0-dim zero (noise_aug_strength is 0.0, so
                             frames_ + 0.0*noise == frames_ bit-for-bit; asserted strength==0 via the call kwargs)
      SK_DIRECT_POST=1       tensor2vid('pil') replaced by the same denormalize on GPU + (x*255).round() + uint8 on
                             GPU, returning HWC uint8 numpy frames (np.array() of them == np.array(PIL image))
      SK_AT_LOAD=path / SK_AT_SAVE=path   pre-fill / dump Triton autotune choices (autotune_cache.py)
      SK_SAVE_LAT=1          save each window's final latents to <SK_OUT>/lat/wNN.pt (timed as its own stage)
      SK_MAX_CHUNKS=N        max_profile_chunks=N (diagnosis runs: stop after N windows)
      SK_REPEAT=K            call ii.run K times in this ONE process (rep k>0 writes to <SK_OUT>/rep<k>)
"""
import importlib.util, os, sys, collections, time, json, hashlib

T_START = time.perf_counter()
T_START_WALL = time.time()
REPO = "/home/kawa/master_project/StereoCrafter"
LANE = f"{REPO}/scripts/distill/runs/more_20261004/pipeline_speed"
sys.path.insert(0, REPO)
os.chdir(REPO)

# import the beyond4 lossless wrapper by path (applies II.write_video_opencv = FFV1 writer)
_spec = importlib.util.spec_from_file_location(
    "infer_lossless", f"{REPO}/scripts/distill/runs/fulldata_v2/beyond4/infer_lossless.py")
_IL = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_IL)

import numpy as np
import torch
import inpainting_inference as ii
import pipelines.mamba_stereo_video_inpainting_pipeline as PM
from diffusers.schedulers.scheduling_euler_discrete import EulerDiscreteScheduler as _ES
from diffusers.utils.torch_utils import randn_tensor

T_IMPORTED = time.perf_counter()

CLIP = os.environ["SK_CLIP"]
OUT = os.environ["SK_OUT"]
STEPS = int(os.environ.get("SK_STEPS", "8"))
GUID = float(os.environ.get("SK_GUID", "1.01"))
UNET = os.environ.get("SK_UNET", "").strip() or None
SIG = [float(x) for x in os.environ.get("SK_SIGMAS", "").split(",") if x.strip()]
PAD_TO = os.environ.get("SK_RNG_PAD_TO", "").strip()
PAD_TO = int(PAD_TO) if PAD_TO else None
RES_H = os.environ.get("SK_H", "").strip()
RES_W = os.environ.get("SK_W", "").strip()
assert bool(RES_H) == bool(RES_W), "set both SK_H and SK_W or neither"
READER = os.environ.get("SK_READER", "tracked").strip() or "tracked"
DECODE_CHUNK = os.environ.get("SK_DECODE_CHUNK", "").strip()
CUDNN_BENCH = os.environ.get("SK_CUDNN_BENCH", "").strip() == "1"
CHANNELS_LAST = [x for x in os.environ.get("SK_CHANNELS_LAST", "").replace("+", ",").split(",") if x.strip()]  # "vae+unet" ok
SKIP_DISCARDED = os.environ.get("SK_SKIP_DISCARDED", "").strip() == "1"
NOAUG_SKIP = os.environ.get("SK_NOAUG_SKIP", "").strip() == "1"
DIRECT_POST = os.environ.get("SK_DIRECT_POST", "").strip() == "1"
AT_LOAD = os.environ.get("SK_AT_LOAD", "").strip()
AT_SAVE = os.environ.get("SK_AT_SAVE", "").strip()
SAVE_LAT = os.environ.get("SK_SAVE_LAT", "").strip() == "1"
MAX_CHUNKS = os.environ.get("SK_MAX_CHUNKS", "").strip()
REPEAT = int(os.environ.get("SK_REPEAT", "1") or "1")
assert set(CHANNELS_LAST) <= {"vae", "unet"}, CHANNELS_LAST
CFG_PATH = "config/0160_overfit_inference_matched.json"
CFG = json.load(open(CFG_PATH))

assert ii.write_video_opencv is _IL._patched_write, "lossless writer patch did not take"
if READER == "lowmem":
    sys.path.insert(0, LANE)
    from lowmem_reader import read_and_prepare_video_lowmem as _lowmem_read
    ii.read_and_prepare_video = _lowmem_read
    print("[sk] reader = lowmem_reader.read_and_prepare_video_lowmem", flush=True)
elif READER == "cropfirst":
    sys.path.insert(0, LANE)
    from reader_cropfirst import read_cropfirst as _crop_read
    _CH = int(RES_H) if RES_H else int(CFG["target_height"])
    _CW = int(RES_W) if RES_W else int(CFG["target_width"])

    def _cropfirst_reader(input_video_path, return_right=False):
        assert not return_right
        return _crop_read(input_video_path, _CH, _CW)      # main()'s _center_crop_frames is then a full-size view

    ii.read_and_prepare_video = _cropfirst_reader
    print(f"[sk] reader = reader_cropfirst.read_cropfirst crop={_CH}x{_CW}", flush=True)
else:
    assert READER == "tracked", READER
    print("[sk] reader = tracked utils.inpainting.read_and_prepare_video", flush=True)

if CUDNN_BENCH:
    torch.backends.cudnn.benchmark = True

# ------------------------------------------------------------------ stage timeline (passive)
TL = []          # (name, t0, t1, extra) with perf_counter times, for the CURRENT repetition
RUN_STATE = dict(window=-1)


def _sync():
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        torch.cuda.synchronize()


def timed(name, fn, extra_fn=None):
    def w(*a, **k):
        _sync()
        t0 = time.perf_counter()
        out = fn(*a, **k)
        _sync()
        t1 = time.perf_counter()
        TL.append((name, t0, t1, dict(window=RUN_STATE["window"], **(extra_fn(a, k, out) if extra_fn else {}))))
        return out
    w.__wrapped_stage__ = fn
    return w


def wrap_cls_method(cls, attr, name):
    orig = getattr(cls, attr)
    setattr(cls, attr, timed(name, orig))


def wrap_classmethod(cls, attr, name):
    orig = getattr(cls, attr)          # bound classmethod
    setattr(cls, attr, staticmethod(timed(name, orig)))


# model load sub-stages
wrap_classmethod(ii.CLIPVisionModelWithProjection, "from_pretrained", "load_clip")
wrap_classmethod(ii.AutoencoderKLTemporalDecoder, "from_pretrained", "load_vae")
wrap_classmethod(ii.UNetSpatioTemporalConditionModel, "from_pretrained", "load_unet")
wrap_classmethod(ii._Pipe, "from_pretrained", "load_pipe")
wrap_cls_method(ii._Pipe, "to", "pipe_to_cuda")
_torch_load = torch.load
torch.load = timed("torch_load_state", _torch_load)

# video read (tracked or lowmem) + center crop
_reader = ii.read_and_prepare_video
_READ_INFO = {}


def _read_extra(a, k, out):
    fps, fl, fw, fm = out[:4]
    _READ_INFO.update(T=int(fw.shape[0]), shape=list(fw.shape))
    return dict(T=int(fw.shape[0]), shape=list(fw.shape))


ii.read_and_prepare_video = timed("read_video", _reader, _read_extra)
ii._center_crop_frames = timed("center_crop", ii._center_crop_frames)

# per-window pipeline internals (class level so they apply to the pipeline main() builds)
wrap_cls_method(ii._Pipe, "_encode_image", "clip_encode")
wrap_cls_method(ii._Pipe, "_encode_vae_frames", "vae_encode")
wrap_cls_method(ii._Pipe, "_encode_mask_frames", "mask_encode")
wrap_cls_method(ii._Pipe, "decode_latents", "vae_decode")

# VaeImageProcessor.preprocess (CPU): image_processor (frames) vs mask_processor (do_convert_grayscale=True)
from diffusers.image_processor import VaeImageProcessor as _VIP
_vip_pre = _VIP.preprocess


def _vip_pre_stage(self, *a, **k):
    name = "mask_preprocess" if getattr(self.config, "do_convert_grayscale", False) else "img_preprocess"
    return timed(name, _vip_pre)(self, *a, **k)


_VIP.preprocess = _vip_pre_stage

# randn_tensor as seen by the pipeline module: CPU call = noise_aug draw, CUDA call = initial latents
_pm_randn = PM.randn_tensor


def _randn_stage(*a, **k):
    dev = k.get("device")
    is_cpu = dev is not None and torch.device(dev).type == "cpu"
    name = "noise_aug_randn_cpu" if is_cpu else "latent_randn_cuda"
    if is_cpu and NOAUG_SKIP:
        # only legal because main() passes noise_aug_strength=0.0 (checked in the window wrapper below)
        assert RUN_STATE.get("noise_aug_strength") == 0.0, RUN_STATE.get("noise_aug_strength")
        name = "noise_aug_SKIPPED"
        fn = lambda *aa, **kk: torch.zeros((), dtype=kk.get("dtype") or torch.float32)
    else:
        fn = _pm_randn
    return timed(name, fn)(*a, **k)


PM.randn_tensor = _randn_stage

# tensor2vid (latent decode output -> PIL) as called by main()
_t2v = ii.tensor2vid


def _t2v_direct(video, processor, output_type="np"):
    assert output_type == "pil", output_type
    outs = []
    for b in range(video.shape[0]):
        batch_vid = video[b].permute(1, 0, 2, 3)
        img = torch.stack([processor.denormalize(batch_vid[i]) for i in range(batch_vid.shape[0])])
        # pt_to_numpy: .cpu().permute(0,2,3,1).float().numpy(); numpy_to_pil: (x*255).round().astype(uint8)
        u8 = (img.permute(0, 2, 3, 1).float() * 255).round().to(torch.uint8).cpu().numpy()
        outs.append([u8[i] for i in range(u8.shape[0])])
    return outs


ii.tensor2vid = timed("tensor2vid_DIRECT" if DIRECT_POST else "tensor2vid_pil", _t2v_direct if DIRECT_POST else _t2v)

# writer (FFV1 encode inside the patched writer) and the md5 it computes
_IL._ffv1_write = timed("ffv1_encode", _IL._ffv1_write)
ii.write_video_opencv = timed("write_call", _IL._patched_write,
                              lambda a, k, out: dict(file=os.path.basename(a[2]), shape=list(a[0].shape)))

# ------------------------------------------------------------------ window plan (main()'s loop, replicated)
def window_plan(T, chunk, overlap):
    plan, generated = [], False
    for i in range(0, T, chunk - overlap):
        if i + overlap >= T:
            break
        if generated and i + chunk > T:
            cur_i = max(T + overlap - chunk, 0)
            cur_overlap = i - cur_i + overlap
        else:
            cur_i, cur_overlap = i, overlap
        plan.append(dict(i=i, cur_i=cur_i, cur_overlap=cur_overlap, discard=0 if i == 0 else cur_overlap))
        generated = True
    return plan


SKIP_LOG = []
if SKIP_DISCARDED:
    _dec = ii._Pipe.decode_latents.__wrapped_stage__        # the original (un-timed) method

    def _decode_skip(self, latents, num_frames, decode_chunk_size=14):
        plan = window_plan(_READ_INFO["T"], int(CFG["frames_chunk"]), int(CFG["overlap"]))
        disc = plan[RUN_STATE["window"]]["discard"]
        if disc == 0:
            return _dec(self, latents, num_frames, decode_chunk_size)
        # identical to decode_latents except chunks whose frames are ALL < disc are not decoded (zeros instead)
        import inspect
        from diffusers.utils.torch_utils import is_compiled_module
        latents = latents.flatten(0, 1)
        latents = 1 / self.vae.config.scaling_factor * latents
        forward_vae_fn = self.vae._orig_mod.forward if is_compiled_module(self.vae) else self.vae.forward
        accepts_num_frames = "num_frames" in set(inspect.signature(forward_vae_fn).parameters.keys())
        frames, skipped = [], 0
        shape_ref = None
        pending = []
        for i in range(0, latents.shape[0], decode_chunk_size):
            num_frames_in = latents[i: i + decode_chunk_size].shape[0]
            if i + num_frames_in <= disc:
                pending.append((len(frames), num_frames_in))
                frames.append(None)
                skipped += 1
                continue
            decode_kwargs = {"num_frames": num_frames_in} if accepts_num_frames else {}
            frame = self.vae.decode(latents[i: i + decode_chunk_size], **decode_kwargs).sample
            shape_ref = frame
            frames.append(frame)
        for pos, n in pending:
            frames[pos] = torch.zeros((n,) + tuple(shape_ref.shape[1:]), dtype=shape_ref.dtype, device=shape_ref.device)
        SKIP_LOG.append(dict(window=RUN_STATE["window"], discard=disc, chunks_skipped=skipped,
                             chunks_total=len(frames)))
        frames = torch.cat(frames, dim=0)
        frames = frames.reshape(-1, num_frames, *frames.shape[1:]).permute(0, 2, 1, 3, 4)
        frames = frames.float()
        return frames

    ii._Pipe.decode_latents = timed("vae_decode", _decode_skip)

# ------------------------------------------------------------------ custom sigma schedule (SK_SIGMAS)  [unchanged]
_orig_set = _ES.set_timesteps
_orig_step = _ES.step
_REF = {}
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

# ------------------------------------------------------------------ RNG pad (SK_RNG_PAD_TO)  [unchanged]
STATS = dict(pad_draws=0, windows=[])
CUR = {}

if PAD_TO is not None:
    def _step_pad(self, model_output, timestep, sample, *a, **k):
        out = _orig_step(self, model_output, timestep, sample, *a, **k)
        if self._step_index == len(self.timesteps):
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

print(f"[sk] clip={CLIP} steps={STEPS} guid={GUID} unet={UNET} out={OUT} "
      f"sigmas={SIG or None} rng_pad_to={PAD_TO} res={(RES_H + 'x' + RES_W) if RES_H else 'config'} "
      f"reader={READER} decode_chunk={DECODE_CHUNK or 'config'} cudnn_bench={int(CUDNN_BENCH)} "
      f"channels_last={CHANNELS_LAST or None} skip_discarded={int(SKIP_DISCARDED)} noaug_skip={int(NOAUG_SKIP)} "
      f"direct_post={int(DIRECT_POST)} at_load={AT_LOAD or None} at_save={AT_SAVE or None} "
      f"save_lat={int(SAVE_LAT)} max_chunks={MAX_CHUNKS or None} repeat={REPEAT}", flush=True)

AT_INFO = {}
if AT_LOAD:
    sys.path.insert(0, LANE)
    import autotune_cache as _atc
    AT_INFO["loaded_entries"] = _atc.load(AT_LOAD)
    print(f"[sk] autotune choices pre-filled from {AT_LOAD}: {AT_INFO['loaded_entries']} entries", flush=True)

# ------------------------------------------------------------------ passive instrumentation  [unchanged + stage]
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
CL_DONE = {}


def wrapped_call(self, *a, **k):
    global T_FIRST_CALL
    if T_FIRST_CALL is None:
        T_FIRST_CALL = time.perf_counter()
    RUN_STATE["noise_aug_strength"] = k.get("noise_aug_strength")
    unet = self.unet
    if CHANNELS_LAST and not CL_DONE.get(id(self)):
        _sync()
        t0 = time.perf_counter()
        # module.to(memory_format=channels_last) raises on the temporal Conv3d weights -> convert Conv2d weights only
        for part in CHANNELS_LAST:
            n_cl = 0
            for m in getattr(self, part).modules():
                if isinstance(m, torch.nn.Conv2d):
                    m.weight.data = m.weight.data.contiguous(memory_format=torch.channels_last)
                    n_cl += 1
            print(f"[sk] channels_last: {part}: {n_cl} Conv2d weights converted", flush=True)
        _sync()
        TL.append(("channels_last_convert", t0, time.perf_counter(), dict(window=RUN_STATE["window"])))
        CL_DONE[id(self)] = True
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
    unet_each = [s.elapsed_time(e) for s, e in evs]
    rec = dict(call_s=t1 - t0, unet_calls=len(evs), batches=CUR.get("batches", []),
               unet_ms=sum(unet_each), unet_ms_each=unet_each,
               init_md5=CUR.get("init_md5"), init_shape=CUR.get("init_shape"), init_dtype=CUR.get("init_dtype"),
               pad_draws=CUR.get("pad_draws", 0), num_frames=k.get("num_frames"))
    STATS["windows"].append(rec)
    TL.append(("pipe_call", t0, t1, dict(window=RUN_STATE["window"], unet_ms=rec["unet_ms"])))
    return out


ii._Pipe.__call__ = wrapped_call

# spatial_tiled_process = one window (encode + sampling); also the latent save point
_stp = ii.spatial_tiled_process


def _stp_wrapped(*a, **k):
    RUN_STATE["window"] += 1
    out = timed("window_sampling", _stp)(*a, **k)
    if SAVE_LAT:
        _sync()
        t0 = time.perf_counter()
        d = os.path.join(RUN_STATE["out"], "lat")
        os.makedirs(d, exist_ok=True)
        torch.save(out.detach().cpu(), os.path.join(d, f"w{RUN_STATE['window']:02d}.pt"))
        TL.append(("save_latents", t0, time.perf_counter(), dict(window=RUN_STATE["window"])))
    return out


ii.spatial_tiled_process = _stp_wrapped

# ------------------------------------------------------------------ run(s)
kw = dict(config=CFG_PATH,
          input_video_path=f"video_data/splatting/{CLIP}_splatting_results.mp4",
          num_inference_steps=STEPS, min_guidance_scale=GUID, max_guidance_scale=GUID)
if RES_H:
    kw["target_height"] = int(RES_H)
    kw["target_width"] = int(RES_W)
    kw["tile_num"] = 1
if UNET is None:
    kw["unet_state_path"] = None
else:
    kw["unet_state_path"] = UNET
    kw["expected_partial_unet_state"] = True
    kw["mamba_gate_override"] = 1.0
if DECODE_CHUNK:
    kw["decode_chunk_size"] = int(DECODE_CHUNK)
if MAX_CHUNKS:
    kw["max_profile_chunks"] = int(MAX_CHUNKS)


def summarize(tl, t_run0, t_run1, windows):
    by = collections.OrderedDict()
    for name, t0, t1, ex in tl:
        d = by.setdefault(name, dict(n=0, s=0.0))
        d["n"] += 1
        d["s"] += t1 - t0
    return by


RUNS = []
for rep in range(REPEAT):
    out_dir = OUT if rep == 0 else os.path.join(OUT, f"rep{rep}")
    RUN_STATE.update(window=-1, out=out_dir)
    TL.clear()
    STATS["windows"] = []
    SKIP_LOG.clear()
    t_run0 = time.perf_counter()
    ii.run(save_dir=out_dir, **kw)
    _sync()
    t_run1 = time.perf_counter()
    tl = list(TL)
    W = list(STATS["windows"])
    stages = summarize(tl, t_run0, t_run1, W)
    md5_lines = open(os.path.join(out_dir, "writer_md5.txt")).read().strip().splitlines()
    RUNS.append(dict(rep=rep, out=out_dir, t_run0=t_run0 - T_START, t_run1=t_run1 - T_START,
                     run_s=t_run1 - t_run0, stages=stages, n_windows=len(W),
                     unet_ms_sum=sum(w["unet_ms"] for w in W), call_s_sum=sum(w["call_s"] for w in W),
                     unet_calls_per_window=sorted(set(w["unet_calls"] for w in W)),
                     batch_sizes=sorted(set(b for w in W for b in w["batches"])),
                     init_md5=[w["init_md5"] for w in W], windows=W, skip_log=list(SKIP_LOG),
                     timeline=[(n, t0 - T_START, t1 - T_START, ex) for n, t0, t1, ex in tl],
                     writer_md5=md5_lines))
    print(f"[stage] rep={rep} run_s={t_run1 - t_run0:.2f} " +
          " ".join(f"{n}={d['s']:.2f}/{d['n']}" for n, d in stages.items()), flush=True)
    print(f"[speed] rep={rep} windows={len(W)} unet_calls={sum(w['unet_calls'] for w in W)} "
          f"per_window={RUNS[-1]['unet_calls_per_window']} batch={RUNS[-1]['batch_sizes']} "
          f"call_s_sum={RUNS[-1]['call_s_sum']:.2f} unet_s_sum={RUNS[-1]['unet_ms_sum'] / 1000:.2f} "
          f"init_md5_all={hashlib.md5(''.join(RUNS[-1]['init_md5']).encode()).hexdigest()}", flush=True)

if AT_SAVE:
    sys.path.insert(0, LANE)
    import autotune_cache as _atc
    AT_INFO["saved"] = _atc.save(AT_SAVE)
    print(f"[sk] autotune choices saved to {AT_SAVE}: "
          f"{sum(len(v) for v in AT_INFO['saved'].values())} entries", flush=True)
else:
    try:
        sys.path.insert(0, LANE)
        import autotune_cache as _atc
        AT_INFO["snapshot"] = _atc.snapshot()
    except Exception as e:  # diagnostic only
        AT_INFO["snapshot_error"] = repr(e)

T_END = time.perf_counter()
summary = dict(clip=CLIP, out=OUT, guid=GUID, steps=STEPS, sigmas=SIG or None, rng_pad_to=PAD_TO,
               unet_state=UNET, res=(RES_H + "x" + RES_W) if RES_H else "config", reader=READER,
               knobs=dict(decode_chunk=DECODE_CHUNK or None, cudnn_bench=CUDNN_BENCH, channels_last=CHANNELS_LAST,
                          skip_discarded=SKIP_DISCARDED, noaug_skip=NOAUG_SKIP, direct_post=DIRECT_POST,
                          at_load=AT_LOAD or None, at_save=AT_SAVE or None, save_lat=SAVE_LAT,
                          max_chunks=MAX_CHUNKS or None, repeat=REPEAT),
               env=dict(TRITON_CACHE_DIR=os.environ.get("TRITON_CACHE_DIR"),
                        TRITON_PRINT_AUTOTUNING=os.environ.get("TRITON_PRINT_AUTOTUNING"),
                        CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES"),
                        torch_threads=torch.get_num_threads()),
               t_start_wall=T_START_WALL, import_s=T_IMPORTED - T_START,
               first_call_s=(T_FIRST_CALL - T_START) if T_FIRST_CALL else None,
               hook_total_s=T_END - T_START, schedule=SCHED_LOG or None, autotune=AT_INFO, runs=RUNS)
with open(os.path.join(OUT, "stage_log.json"), "w") as fh:
    json.dump(summary, fh, indent=1, default=str)
print(f"[stage] import_s={summary['import_s']:.2f} hook_total_s={summary['hook_total_s']:.2f}", flush=True)
print("[sk] done", flush=True)
