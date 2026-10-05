#!/usr/bin/env python
"""M2SVid on StereoCrafter's DEPLOYED inputs -> lossless FFV1 SBS renders (deep_20261004 / external_models lane).

Definitions and gates: PREREG_r2.txt (same dir).  Nothing tracked is modified.  The M2SVid repo (google-research/m2svid
@11b0133), weights and venv live under /mnt/ssd_data/deep_20261004/external_models/.  One model load, then the listed
clips in order.  Inputs are read exactly like utils/inpainting.read_and_prepare_video + _center_crop_frames (576x1024
window of the 2x2 splatting tile), then M2SVid's official preprocessing (inpaint_and_refine.py), then
VideoLDM.generate() per window, then the official uint8 conversion.  Output = the model's fully generated right eye.

usage (GPU 0 under its lock, lane venv):
  CUDA_VISIBLE_DEVICES=0 flock /tmp/claude-gpu0.lock /mnt/ssd_data/deep_20261004/external_models/venv/bin/python \
      m2svid_infer_ll.py --tag m2svid_fa_w16 --win 16 --clips 0301 0204
  --smoke DIR : first window of the first clip only; writes DIR/smoke_<clip>.json + png, no video.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

LANE = "/mnt/ssd_data/deep_20261004/external_models"
M2 = f"{LANE}/m2svid"
REPO = "/home/kawa/master_project/StereoCrafter"
REC = f"{REPO}/scripts/distill/runs/deep_20261004/external_models"
OUT_ROOT = f"{REPO}/outputs/deep_20261004/external_models"
FFMPEG = "/home/kawa/miniconda3/envs/stereocrafter/bin/ffmpeg"
TH, TW = 576, 1024

ap = argparse.ArgumentParser()
ap.add_argument("--clips", nargs="+", required=True)
ap.add_argument("--tag", default="m2svid_fa_w16")
ap.add_argument("--win", type=int, default=16)
ap.add_argument("--seed", type=int, default=1234)
ap.add_argument("--config", default=f"{REC}/m2svid_fa_infer.yaml")
ap.add_argument("--ckpt", default=f"{LANE}/ckpts/m2svid_weights.pt")
ap.add_argument("--smoke", default=None)
ap.add_argument("--identity_guider", action="store_true", help="fallback F1 (PREREG_r2), only after an OOM")
ap.add_argument("--mask_thresh", type=float, default=None, help="DEV-only variant (PREREG_r2 secondary): binarize the "
                "soft StereoCrafter occlusion map at this value BEFORE the official closing (None = official path)")
ap.add_argument("--decode_chunk", type=int, default=0, help="fallback F2 (PREREG_r2): VAE-decode N frames at a time "
                "(0 = official whole-window decode)")
args = ap.parse_args()

os.environ.setdefault("TORCH_HOME", f"{LANE}/torch_home")
os.environ.setdefault("HF_HOME", f"{LANE}/hf_home")
sys.path[:0] = [M2, f"{M2}/third_party/Hi3D-Official"]
os.chdir(M2)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from decord import VideoReader, cpu  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from pytorch_lightning import seed_everything  # noqa: E402
from torchvision import transforms  # noqa: E402

import m2svid.models_for_sgm.m2svid_model as MM  # noqa: E402


class _NoLPIPS(torch.nn.Module):
    """VideoLDM builds sgm's VGG-LPIPS for test_step metrics only (no state-dict entries).  Stubbed: no downloads."""

    def forward(self, *a, **k):
        raise RuntimeError("LPIPS metric is stubbed in this inference driver")


MM.LPIPS = _NoLPIPS
from sgm.util import instantiate_from_config  # noqa: E402
from m2svid.data.utils import apply_closing, apply_dilation  # noqa: E402

T0 = time.time()


def log(*a):
    print(f"[m2svid {time.time() - T0:7.1f}s]", *a, flush=True)


def ffv1_write(arr, fps, path):
    """verbatim from scripts/distill/runs/fulldata_v2/beyond4/infer_lossless.py::_ffv1_write"""
    if arr.dtype != np.uint8:
        raise TypeError(f"lossless writer expects uint8, got {arr.dtype}")
    T, H, W, C = arr.shape
    cmd = [FFMPEG, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", f"{float(fps):.6f}", "-i", "-", "-an", "-c:v", "ffv1", "-level", "3", "-g", "1", "-slicecrc", "1",
           "-threads", "8", "-pix_fmt", "bgr0", path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(T):
        proc.stdin.write(np.ascontiguousarray(arr[i]).tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg FFV1 encode failed: {' '.join(cmd)}")


def read_inputs(clip):
    """utils/inpainting.read_and_prepare_video + inpainting_inference._center_crop_frames, cropped in uint8 first
    (elementwise ops -> identical values), returns uint8 [T,576,1024,3] left / mask-tile / warped + fps."""
    vr = VideoReader(f"{REPO}/video_data/splatting/{clip}_splatting_results.mp4", ctx=cpu(0))
    fps = float(vr.get_avg_fps())
    T = len(vr)
    f0 = vr[0].asnumpy()
    H2, W2 = f0.shape[0] // 2, f0.shape[1] // 2
    h128, w128 = H2 // 128 * 128, W2 // 128 * 128
    top, left = (h128 - TH) // 2, (w128 - TW) // 2
    L = np.empty((T, TH, TW, 3), np.uint8)
    M = np.empty_like(L)
    B = np.empty_like(L)
    for s in range(0, T, 8):
        idx = list(range(s, min(s + 8, T)))
        b = vr.get_batch(idx).asnumpy()
        L[s:s + len(idx)] = b[:, top:top + TH, left:left + TW]
        M[s:s + len(idx)] = b[:, H2 + top:H2 + top + TH, left:left + TW]
        B[s:s + len(idx)] = b[:, H2 + top:H2 + top + TH, W2 + left:W2 + left + TW]
    return fps, T, (top, left, H2, W2), L, M, B


def prepare(L, M, B):
    """origin's float32 tensors, then M2SVid's official preprocessing (inpaint_and_refine.py, --mask_antialias 0)."""
    left_f = torch.from_numpy(L).permute(0, 3, 1, 2).float() / 255.0
    warped_f = torch.from_numpy(B).permute(0, 3, 1, 2).float() / 255.0
    mask_f = (torch.from_numpy(M).permute(0, 3, 1, 2).float() / 255.0).mean(dim=1, keepdim=True)
    left_sbs = (left_f * 255).permute(0, 2, 3, 1).to(dtype=torch.uint8).numpy()   # origin's SBS left half
    pol = dict(  # mask polarity: M2SVid expects mask>0.5 == hole (warped black there)
        hole_frac=float((mask_f > 0.5).float().mean()),
        warped_dark_in_hole=float((warped_f.amax(1, keepdim=True)[mask_f > 0.5] < 10 / 255).float().mean())
        if bool((mask_f > 0.5).any()) else None,
        warped_dark_outside=float((warped_f.amax(1, keepdim=True)[mask_f <= 0.5] < 10 / 255).float().mean()))
    if args.mask_thresh is not None:   # DEV-only variant: approximate M2SVid's own coverage==0 mask
        mask_f = (mask_f > args.mask_thresh).float()
        pol["hole_frac_binarized"] = float(mask_f.mean())
    mask = apply_closing(mask_f.clone(), 11)
    warped = warped_f.clone()
    warped[mask.repeat(1, 3, 1, 1) > 0.5] = 0
    mask = apply_dilation(mask, 3)
    mask = mask.repeat(1, 3, 1, 1)
    vid_left = left_f.permute(1, 0, 2, 3).float() * 2 - 1        # [c,t,h,w]
    vid_warp = warped.permute(1, 0, 2, 3).float() * 2 - 1
    vid_mask = mask.permute(1, 0, 2, 3).float() * 2 - 1
    vid_mask = vid_mask.permute(1, 0, 2, 3).float()
    vid_mask = transforms.Resize([TH // 8, TW // 8], antialias=0)(vid_mask)
    vid_mask = vid_mask[:, [0]].permute(1, 0, 2, 3).float()      # [1,t,h/8,w/8]
    pol["hole_frac_after_closing_dilation"] = float((mask[:, 0] > 0.5).float().mean())
    return left_sbs, vid_left, vid_warp, vid_mask, pol


def windows(T, win):
    """non-overlapping windows; the last one anchored to the clip end, only its new frames kept."""
    if T <= win:
        return [(0, T, 0)]
    out = [(s, s + win, 0) for s in range(0, T - win + 1, win)]
    end = out[-1][1]
    if end < T:
        out.append((T - win, T, end - (T - win)))
    return out


@torch.no_grad()
def generate_identity(model, batch):
    """Fallback F1 only: VideoLDM.generate with the scale-1.0 LinearPredictionGuider replaced by its conditional half."""
    from einops import rearrange
    from sgm.modules.diffusionmodules.guiders import IdentityGuider
    batch = model.add_custom_cond(batch, infer=True)
    frames = model.get_input(batch)
    N = len(frames)
    keys = [e.input_key for e in model.conditioner.embedders]
    c, uc = model.conditioner.get_unconditional_conditioning(batch, force_uc_zero_embeddings=keys)
    x = rearrange(frames, "b c t h w -> (b t) c h w").to(model.device)
    extra = {"image_only_indicator": torch.zeros(N, batch["num_video_frames"]).to(model.device),
             "num_video_frames": batch["num_video_frames"], "inpainting_mask": batch["inpainting_mask"]}

    def denoiser(inp, sigma, cc):
        return model.denoiser(model.model, inp, sigma, cc, **extra)

    old = model.sampler.guider
    model.sampler.guider = IdentityGuider()
    try:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            shape = (x.shape[0], 4, int(x.shape[2] // 8), int(x.shape[3] // 8))
            samples = model.sampler(denoiser, torch.randn(shape, device=model.device), cond=c, uc=uc,
                                    num_video_frames=batch["num_video_frames"])
    finally:
        model.sampler.guider = old
    samples = model.decode_first_stage(samples.half(), num_video_frames=batch["num_video_frames"])
    return rearrange(samples, "(b t) c h w -> b c t h w", t=batch["num_video_frames"])


def run_window(model, vid_left, vid_warp, vid_mask, fps, s, e):
    batch = {
        "video": vid_left[:, s:e].contiguous()[None].cuda(),
        "video_2nd_view": vid_left[:, s:e].contiguous()[None].cuda(),
        "reprojected_video": vid_warp[:, s:e].contiguous()[None].cuda(),
        "reprojected_mask": vid_mask[:, s:e].contiguous()[None].cuda(),
        "fps_id": torch.tensor([fps]).cuda(),
        "caption": [""],
        "motion_bucket_id": torch.tensor([127]).cuda(),
    }
    torch.cuda.synchronize()
    t = time.time()
    with torch.inference_mode():
        if args.identity_guider:
            gen = generate_identity(model, batch)
        else:
            gen = model.generate(batch)["generated-video"]
    torch.cuda.synchronize()
    dt = time.time() - t
    g = gen[0].cpu().numpy().transpose(1, 2, 3, 0)                 # [t,h,w,3], float16 as the official writer sees it
    finite = bool(np.isfinite(g.astype(np.float32)).all())
    g8 = (((g + 1) / 2).clip(0, 1) * 255).astype(np.uint8)          # inpaint_and_refine.save_video conversion
    return g8, dt, finite, str(g.dtype)


def origin_ll(clip):
    p = f"{REPO}/outputs/beyond4_lossless/clips/{clip}_origin_ll/{clip}_inpainting_results_sbs.mkv"
    vr = VideoReader(p, ctx=cpu(0))
    v = vr.get_batch(list(range(len(vr)))).asnumpy()
    return p, v[:, :, :TW], v[:, :, TW:]


def half(a):
    return cv2.resize(a, (TW // 2, TH // 2), interpolation=cv2.INTER_AREA)


def panel(fi, tiles, labels, path):
    img = np.concatenate([half(a) for a in tiles], 1)
    for q, s in enumerate(labels):
        cv2.putText(img, s, (q * (TW // 2) + 6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(img, f"frame {fi}", (6, TH // 2 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


# ------------------------------------------------------------------------------------------------ model
log(f"args {vars(args)}")
seed_everything(args.seed)
config = OmegaConf.load(args.config)
model = instantiate_from_config(config.model).cpu()
sd_raw = torch.load(args.ckpt, map_location="cpu")
raw_keys = list(sd_raw["module"].keys())
sd = {k[len("module."):]: sd_raw["module"][k] for k in raw_keys}     # == VideoLDM.init_from_ckpt (.pt branch)
missing, unexpected = model.load_state_dict(sd, strict=False)
g2 = dict(n_ckpt=len(raw_keys), n_ckpt_module_prefix=sum(k.startswith("module.") for k in raw_keys),
          n_model=len(model.state_dict()), missing=list(missing), unexpected=list(unexpected),
          unexpected_prefixes=sorted({k.split(".")[0] for k in unexpected}),
          G2_pass=len(missing) == 0 and all(k.startswith("loss_fn.") for k in unexpected))
log(f"G2 restore: ckpt {g2['n_ckpt']} keys, model {g2['n_model']}, missing {len(missing)}, unexpected "
    f"{len(unexpected)} {g2['unexpected_prefixes']} -> G2_pass={g2['G2_pass']}")
del sd_raw, sd
model = model.cuda().half().eval()
log(f"model on GPU, allocated {torch.cuda.memory_allocated() / 2**30:.2f} GiB; sampler {type(model.sampler).__name__} "
    f"guider {type(model.sampler.guider).__name__} steps {model.sampler.num_steps}")

if args.decode_chunk > 0:
    from sgm.util import default  # noqa: F401  (same helper the original uses)

    def _decode_chunked(z, num_video_frames=None):
        """F2: sgm DiffusionEngine.decode_first_stage (VideoDecoder branch) with the whole-window decode split into
        chunks of --decode_chunk frames (temporal decoder context = chunk).  Everything else identical."""
        z = 1.0 / model.scale_factor * z
        outs = []
        with torch.autocast("cuda", enabled=not model.disable_first_stage_autocast):
            for s0 in range(0, z.shape[0], args.decode_chunk):
                zz = z[s0:s0 + args.decode_chunk]
                outs.append(model.first_stage_model.decode(zz, timesteps=len(zz)))
        return torch.cat(outs, dim=0)

    model.decode_first_stage = _decode_chunked
    log(f"F2 active: VAE decode in chunks of {args.decode_chunk} frames")

# ------------------------------------------------------------------------------------------------ clips
for ci, clip in enumerate(args.clips):
    tc = time.time()
    seed_everything(args.seed)
    fps, T, geo, L, M, B = read_inputs(clip)
    left_sbs, vid_left, vid_warp, vid_mask, pol = prepare(L, M, B)
    wins = windows(T, args.win)
    log(f"{clip}: {T} frames fps {fps:.4f} window(top,left,H2,W2)={geo} -> {len(wins)} windows of {args.win}; "
        f"mask polarity {pol}")
    if args.smoke:
        os.makedirs(args.smoke, exist_ok=True)
        js = os.path.join(args.smoke, f"smoke_{clip}_{args.tag}.json")
        assert not os.path.exists(js), f"refusing to overwrite {js}"
        s, e, k0 = wins[0]
        torch.cuda.reset_peak_memory_stats()
        g8, dt, finite, gdt = run_window(model, vid_left, vid_warp, vid_mask, fps, s, e)
        peak = torch.cuda.max_memory_allocated() / 2**30
        peak_r = torch.cuda.max_memory_reserved() / 2**30
        # second pass of the same window: timing without first-call overhead
        g8b, dt2, _, _ = run_window(model, vid_left, vid_warp, vid_mask, fps, s, e)
        po, Lo, Ro = origin_ll(clip)
        fi = (e - s) // 2
        panel(fi, [L[s + fi], B[s + fi], M[s + fi], Ro[s + fi], g8[fi]],
              ["left (input)", "warped (input)", "mask tile", "origin right", f"M2SVid right ({args.tag})"],
              os.path.join(args.smoke, f"smoke_{clip}_{args.tag}_f{s + fi:03d}.png"))
        out = dict(clip=clip, tag=args.tag, window=[s, e], seconds_first=dt, seconds_second=dt2, peak_alloc_GiB=peak,
                   peak_reserved_GiB=peak_r, finite=finite, gen_dtype=gdt, out_std=float(g8.std()),
                   rerun_identical=bool((g8 == g8b).all()), rerun_maxabs=int(np.abs(g8.astype(int) - g8b).max()),
                   left_md5_window=hashlib.md5(np.ascontiguousarray(left_sbs[s:e]).tobytes()).hexdigest(),
                   origin_left_md5_window=hashlib.md5(np.ascontiguousarray(Lo[s:e]).tobytes()).hexdigest(),
                   mask_polarity=pol, G2=g2, fps=fps, geo=geo, identity_guider=args.identity_guider)
        json.dump(out, open(js, "w"), indent=1)
        log(f"SMOKE {clip} win {s}-{e}: {dt:.2f}s (2nd {dt2:.2f}s) peak alloc {peak:.2f} GiB reserved {peak_r:.2f} GiB "
            f"finite {finite} std {out['out_std']:.2f} rerun identical {out['rerun_identical']} "
            f"(max |d| {out['rerun_maxabs']}) left md5 match {out['left_md5_window'] == out['origin_left_md5_window']}")
        break

    od = f"{OUT_ROOT}/clips/{clip}_{args.tag}_ll"
    mkv = f"{od}/{clip}_inpainting_results_sbs.mkv"
    assert not os.path.exists(od), f"refusing to overwrite {od}"
    os.makedirs(od)
    R = np.empty((T, TH, TW, 3), np.uint8)
    secs, finite_all = [], True
    torch.cuda.reset_peak_memory_stats()
    for (s, e, k0) in wins:
        g8, dt, finite, gdt = run_window(model, vid_left, vid_warp, vid_mask, fps, s, e)
        R[s + k0:e] = g8[k0:]
        secs.append(dt)
        finite_all &= finite
    peak = torch.cuda.max_memory_allocated() / 2**30
    sbs = np.concatenate([left_sbs, R], axis=2)
    digest = hashlib.md5(np.ascontiguousarray(sbs).tobytes()).hexdigest()
    ffv1_write(sbs, fps, mkv)
    with open(mkv + ".md5", "w") as fh:
        fh.write(f"{digest}  {tuple(sbs.shape)}  fps={float(fps):.6f}\n")
    po, Lo, Ro = origin_ll(clip)
    md5_mine = hashlib.md5(np.ascontiguousarray(left_sbs).tobytes()).hexdigest()
    md5_orig = hashlib.md5(np.ascontiguousarray(Lo).tobytes()).hexdigest()
    rd = VideoReader(mkv, ctx=cpu(0))
    back = rd.get_batch(list(range(len(rd)))).asnumpy()
    lossless = bool(back.shape == sbs.shape and (back == sbs).all())
    fi = T // 2 if T // 2 < len(Ro) else len(Ro) - 1
    panel(fi, [L[fi], B[fi], Ro[fi], R[fi]], ["left (input)", "warped (input)", "origin right", f"M2SVid right"],
          f"{od}/{clip}_panel_f{fi:03d}.png")
    meta = dict(clip=clip, tag=args.tag, mkv=mkv, T=T, fps=fps, geo=geo, windows=wins, sec_per_window=secs,
                sec_clip=time.time() - tc, peak_alloc_GiB=peak, finite=finite_all, right_std=float(R.std()),
                sbs_md5=digest, lossless_roundtrip=lossless, G0_left_md5=md5_mine, origin_left_md5=md5_orig,
                G0_left_md5_match=md5_mine == md5_orig, G1_frames_match=len(Lo) == T, origin_path=po,
                mask_polarity=pol, G2=g2, seed=args.seed, win=args.win, identity_guider=args.identity_guider,
                m2svid_commit="11b0133093d6abfcc6ff953890edf05457975318",
                weights_md5="b8618e8d8995042a72038f45c356f2f6", config=args.config)
    json.dump(meta, open(f"{od}/run_{clip}.json", "w"), indent=1)
    log(f"DONE {clip}: {len(wins)} windows, {np.mean(secs):.2f} s/window (total gen {sum(secs):.1f}s), peak "
        f"{peak:.2f} GiB, finite {finite_all}, lossless {lossless}, G0 left md5 match {md5_mine == md5_orig}, "
        f"frames {T} vs origin {len(Lo)} -> {mkv}")
log("ALL_DONE")
