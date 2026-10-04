"""CONTROL A (clean self-consistency): copy of scripts/distill/runs/diag_trainer/minift/xcheck_mini_ft.py (2026-09-29) with ONE
functional change -- the "null" target is read from the LOSSLESS FFV1 render of the same deployed origin output instead of the
cv2-mp4v render (same pre-encode array, md5 2e533d7755c950d2fc95043f6fb0a51d; beyond4/FAITHFULNESS.txt), so the two runs differ
only in the target's codec.  Seed, eps/sigma sequence, windows, steps, LR, clipping, checkpoints: identical to the original.
Additive diagnostics only (no effect on the optimisation): (a) the left half of the lossless sbs must be BIT-IDENTICAL to the
splatting input's TL crop (the original's "< 3/255" train-tile check is kept as well); (b) the bf16 start point of the 15 trained
tensors is saved to start_bf16.pt so rel ||dW||/||W|| can be measured against the true start (not the fp16 file).
  variant null : target = RIGHT half of outputs/beyond4_lossless/clips/0301_origin_ll/0301_inpainting_results_sbs.mkv (origin's own output, lossless)
  variant pos  : (unchanged) target = TR quadrant (real right eye) of video_data/train/0301_train.mp4, same registered 576x1024 crop
usage: [MINIFT_SIGMAS=deploy8|idx:0,1|700,7.28|lognormal:0.7:1.6] CUDA_VISIBLE_DEVICES=<g> python xcheck_mini_ft_ll.py <null|pos> [out_subdir]
Writes clean_controls/selfA/<out_subdir>/{train_log.csv, meta.json, start_bf16.pt, step100.pt, step200.pt, step300.pt}; never overwrites an existing dir.
"""
import os, sys, json, time, csv, math
REPO = "/home/kawa/master_project/StereoCrafter"; os.chdir(REPO); sys.path.insert(0, REPO)
os.environ.setdefault("MAMBA_SELF_ATTN_INCLUDE", "__nomatch__")
import torch, torch.nn.functional as F
from decord import VideoReader, cpu
from transformers import CLIPVisionModelWithProjection
from diffusers.models.unets.unet_spatio_temporal_condition import UNetSpatioTemporalConditionModel
from diffusers.models.autoencoders.autoencoder_kl_temporal_decoder import AutoencoderKLTemporalDecoder
from diffusers import EulerDiscreteScheduler
from pipelines.stereo_video_inpainting import StableVideoDiffusionInpaintingPipeline as _Pipe
from utils.training_pipeline import configure_unet_memory_features

VARIANT = sys.argv[1]; assert VARIANT in ("null", "pos")
SUB = sys.argv[2] if len(sys.argv) > 2 else VARIANT
CLIP = "0301"; NF = 14; STRIDE = 11; STEPS = 300; LR = 1e-5; SEED = 1234; SAVE_AT = (100, 200, 300)
ORIGIN_SBS = f"outputs/beyond4_lossless/clips/{CLIP}_origin_ll/{CLIP}_inpainting_results_sbs.mkv"   # CONTROL A: LOSSLESS (was outputs/fulldata_v2/clips/{CLIP}_origin/{CLIP}_inpainting_results_sbs.mp4)
ORIGIN_SBS_MP4V_OLD = f"outputs/fulldata_v2/clips/{CLIP}_origin/{CLIP}_inpainting_results_sbs.mp4"
BASE = "scripts/distill/runs/clean_controls/selfA"
OUT = os.path.join(BASE, SUB); k = 1
while os.path.exists(OUT): k += 1; OUT = os.path.join(BASE, f"{SUB}_{k}")
os.makedirs(OUT); print("OUT", OUT, flush=True)

dev = torch.device("cuda:0"); dt = torch.bfloat16
PREF = ["up_blocks.3.attentions.0.transformer_blocks.0.attn1.", "up_blocks.3.attentions.1.transformer_blocks.0.attn1.",
        "up_blocks.3.attentions.2.transformer_blocks.0.attn1."]
pre = "weights/stable-video-diffusion-img2vid-xt-1-1/"; unet_path = "weights/StereoCrafter/"
image_encoder = CLIPVisionModelWithProjection.from_pretrained(pre, subfolder="image_encoder", variant="fp16", torch_dtype=dt)
vae = AutoencoderKLTemporalDecoder.from_pretrained(pre, subfolder="vae", variant="fp16", torch_dtype=dt)
unet = UNetSpatioTemporalConditionModel.from_pretrained(unet_path, subfolder="unet_diffusers", low_cpu_mem_usage=True, torch_dtype=dt)
pipe = _Pipe.from_pretrained(pre, image_encoder=image_encoder, vae=vae, unet=unet, torch_dtype=dt).to(dev)
pipe.vae.eval(); pipe.image_encoder.eval()
for n, p in pipe.unet.named_parameters(): p.requires_grad_(any(n.startswith(q) for q in PREF) and "origin_attn" not in n)
TRAIN = [(n, p) for n, p in pipe.unet.named_parameters() if p.requires_grad]
assert len(TRAIN) == 15, [n for n, _ in TRAIN]
print("trainable", len(TRAIN), [n.split("transformer_blocks.0.")[1] for n, _ in TRAIN][:5], "...", flush=True)
configure_unet_memory_features(pipeline=pipe, enable_gradient_checkpointing=True, checkpoint_use_reentrant=False, attn_mode="auto", ff_chunk_size=None, ff_chunk_dim=1)
pipe.unet.train()
# fp32 master copies + AdamW
master = {n: torch.nn.Parameter(p.detach().float().clone()) for n, p in TRAIN}
torch.save({n: p.detach().cpu().clone() for n, p in TRAIN}, os.path.join(OUT, "start_bf16.pt"))   # CONTROL A diagnostic: exact bf16 start point
opt = torch.optim.AdamW([master[n] for n, _ in TRAIN], lr=LR, betas=(0.9, 0.999), weight_decay=0.0)
sched = EulerDiscreteScheduler.from_config(pipe.scheduler.config); sched.set_timesteps(8, device=dev)
SIGMAS = [float(s) for s in sched.sigmas[:8]]; TS = [float(t) for t in sched.timesteps[:8]]
print("deployment sigmas", [round(s, 3) for s in SIGMAS], "timesteps", [round(t, 3) for t in TS], flush=True)
# ---- training sigma set (env MINIFT_SIGMAS): "deploy8" (default, uniform over the 8 deployment sigmas, P1 behaviour) |
#      "idx:0,1,2" (uniform over those indices of the 8) | "700,286.5,7.28" (values, snapped to the nearest deployment sigma) |
#      "lognormal:0.7:1.6" (ln sigma ~ N(P_mean, P_std) clipped to [0.002, 700], t = 0.25 ln sigma as the trainer / continuous scheduler) ----
SIGSPEC = os.environ.get("MINIFT_SIGMAS", "deploy8")
if SIGSPEC == "deploy8": SIG_MODE, SIG_SUB, LN_P = "deploy8", list(range(8)), None
elif SIGSPEC.startswith("lognormal:"): SIG_MODE, SIG_SUB, LN_P = "lognormal", [], tuple(float(x) for x in SIGSPEC.split(":")[1:3])
elif SIGSPEC.startswith("idx:"): SIG_MODE, SIG_SUB, LN_P = "subset", [int(x) for x in SIGSPEC[4:].split(",")], None
else: SIG_MODE, SIG_SUB, LN_P = "subset", [min(range(8), key=lambda i: abs(math.log(SIGMAS[i]) - math.log(float(v)))) for v in SIGSPEC.split(",")], None
print("sigma spec", SIGSPEC, "->", SIG_MODE, [round(SIGMAS[i], 3) for i in SIG_SUB] if SIG_SUB else LN_P, flush=True)
def draw_sigma(rng):
    if SIG_MODE == "deploy8": idx = int(torch.randint(0, 8, (1,), generator=rng)); return idx, SIGMAS[idx], TS[idx]
    if SIG_MODE == "subset": idx = SIG_SUB[int(torch.randint(0, len(SIG_SUB), (1,), generator=rng))]; return idx, SIGMAS[idx], TS[idx]
    ln_s = LN_P[0] + LN_P[1] * float(torch.randn((1,), generator=rng)); sigma = min(max(math.exp(ln_s), 0.002), 700.0); return -1, sigma, 0.25 * math.log(sigma)

def crop_quadrants(fr):
    """fr: [T,3,H2,W2] float 2x2 tile -> (BR, BL gray mask, TR, TL) at the registered 576x1024 crop (utils/inpainting.py:147-158 + inpainting_inference.py:258-262)."""
    H, W = fr.shape[2] // 2, fr.shape[3] // 2
    TL, TR, BL, BR = fr[:, :, :H, :W], fr[:, :, :H, W:], fr[:, :, H:, :W], fr[:, :, H:, W:]
    h, w = H // 128 * 128, W // 128 * 128
    top, left = (h - 576) // 2, (w - 1024) // 2
    sl = (slice(None), slice(None), slice(top, top + 576), slice(left, left + 1024))
    return BR[:, :, :h, :w][sl], BL[:, :, :h, :w][sl].mean(dim=1, keepdim=True), TR[:, :, :h, :w][sl], TL[:, :, :h, :w][sl]
def read_tile(s, e):
    vr = VideoReader(f"video_data/train/{CLIP}_train.mp4", ctx=cpu(0))
    return torch.from_numpy(vr.get_batch(list(range(s, e))).asnumpy()).permute(0, 3, 1, 2).float() / 255.0
@torch.no_grad()
def encode(cond, mask, target):
    cond, mask, target = cond.to(dev, dt), mask.to(dev, dt), target.to(dev, dt); H, W = cond.shape[2], cond.shape[3]
    emb = pipe._encode_image(cond[0:1].float(), device=dev, num_videos_per_prompt=1, do_classifier_free_guidance=False)
    fc = pipe.image_processor.preprocess(cond, height=H, width=W)
    lat = torch.cat([pipe.vae.encode(fc[i:i + 1]).latent_dist.mode() for i in range(fc.shape[0])], 0).unsqueeze(0)   # RAW cond latents (x1.0), as the pipeline
    fm = pipe.mask_processor.preprocess(mask, height=H, width=W); ml = F.interpolate(fm, scale_factor=1 / pipe.vae_scale_factor).unsqueeze(0).to(dt)
    ft = pipe.image_processor.preprocess(target, height=H, width=W)
    x0 = torch.cat([pipe.vae.encode(ft[i:i + 1]).latent_dist.mode() for i in range(ft.shape[0])], 0).unsqueeze(0) * pipe.vae.config.scaling_factor
    return emb, lat.to(dt), ml, x0.to(dt), torch.tensor([[6.0, 127.0, 0.0]], dtype=dt, device=dev)

# ---- targets / windows ----
vr_sbs = VideoReader(ORIGIN_SBS, ctx=cpu(0)); n_frames = len(vr_sbs); assert n_frames == len(VideoReader(f"video_data/train/{CLIP}_train.mp4", ctx=cpu(0)))
starts = list(range(0, n_frames - NF + 1, STRIDE))   # deployed grid (inpainting_inference.py:268, frames_chunk 14 / overlap 3); the deployed tail window at 140 is only 11 frames long ([140:154]) and is not used
print("windows", len(starts), starts, flush=True)

# ---- sanity: registration of the origin sbs output with the registered crop (frame 30) ----
f30 = read_tile(30, 31); BR30, M30, TR30, TL30 = crop_quadrants(f30)
sbs30 = torch.from_numpy(vr_sbs[30].asnumpy()).permute(2, 0, 1).float().unsqueeze(0) / 255.0
L30, R30 = sbs30[:, :, :, :1024], sbs30[:, :, :, 1024:]
mad_left = (L30 - TL30).abs().mean().item(); mad_right_gt = (R30 - TR30).abs().mean().item(); mad_right_cond = (R30 - BR30).abs().mean().item()
print(f"[sanity] frame30 MAD sbs-left vs TL crop = {mad_left*255:.3f}/255 (codec-only expected < 3); sbs-right(origin out) vs TR(real GT) = {mad_right_gt*255:.2f}/255; vs BR(cond) = {mad_right_cond*255:.2f}/255", flush=True)
assert mad_left < 3 / 255, f"origin sbs not pixel-aligned with the registered crop: MAD {mad_left*255:.2f}/255"
# CONTROL A diagnostic: the lossless sbs left half is the pipeline's pass-through of the SPLATTING input's TL crop -> must be bit-identical (maxabs == 0)
spl30 = torch.from_numpy(VideoReader(f"video_data/splatting/{CLIP}_splatting_results.mp4", ctx=cpu(0))[30].asnumpy()).permute(2, 0, 1).float().unsqueeze(0) / 255.0
_, _, _, TLs30 = crop_quadrants(spl30); maxabs_spl = (L30 - TLs30).abs().max().item() * 255; mad_spl = (L30 - TLs30).abs().mean().item() * 255
print(f"[sanity-A] frame30 MAD sbs-left vs SPLATTING TL crop = {mad_spl:.4f}/255, maxabs = {maxabs_spl:.4f}/255 (lossless: must be exactly 0)", flush=True)
assert maxabs_spl == 0.0, f"lossless sbs left half is not bit-identical to the splatting TL crop: maxabs {maxabs_spl}/255"
if VARIANT == "null":
    tgt30 = R30
    mad_tt = (tgt30 - R30).abs().mean().item(); print(f"[sanity] P1-null target frame30 vs origin output frame30 MAD = {mad_tt*255:.3f}/255", flush=True)
    assert mad_tt < 3 / 255

WINS = []; mask_fracs = []
for s in starts:
    tile = read_tile(s, s + NF); BR, M, TR, TL = crop_quadrants(tile)
    if VARIANT == "null":
        sbs = torch.from_numpy(vr_sbs.get_batch(list(range(s, s + NF))).asnumpy()).permute(0, 3, 1, 2).float() / 255.0
        target = sbs[:, :, :, 1024:]
        # every window: left half of sbs must equal the TL crop (registration), codec only
        assert (sbs[:, :, :, :1024] - TL).abs().mean().item() < 3 / 255
    else:
        target = TR
    mf = M.mean().item(); mask_fracs.append(mf)
    emb, lat, ml, x0, add = encode(BR, M, target)
    WINS.append((s, emb, lat, ml, x0, add))
    del tile, BR, M, TR, TL, target
assert min(mask_fracs) > 0, mask_fracs
print("[sanity] mask fraction per window (inside 576x1024 crop):", [round(m, 4) for m in mask_fracs], flush=True)
print(f"[sanity] latents: lat {tuple(WINS[0][2].shape)} x0 {tuple(WINS[0][4].shape)} mask-latent mean {WINS[0][3].mean().item():.4f}; encode peak {torch.cuda.max_memory_allocated()/2**30:.2f} GiB", flush=True)
torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()

json.dump({"variant": VARIANT, "clip": CLIP, "target": (ORIGIN_SBS + " right half") if VARIANT == "null" else f"video_data/train/{CLIP}_train.mp4 TR quadrant",
           "trainable": [n for n, _ in TRAIN], "nf": NF, "stride": STRIDE, "windows": starts, "steps": STEPS, "lr": LR, "betas": [0.9, 0.999], "wd": 0.0,
           "clip_grad_norm": 1.0, "seed": SEED, "sigmas": SIGMAS, "timesteps": TS, "sigma_spec": SIGSPEC, "cond_latent_scale": 1.0, "x0_scale": pipe.vae.config.scaling_factor,
           "mask_fracs": mask_fracs, "sanity_mad_left_255": mad_left * 255, "sanity_mad_right_vs_gt_255": mad_right_gt * 255,
           "sanity_mad_right_vs_cond_255": mad_right_cond * 255,
           "control": "A clean self-consistency (lossless target)", "target_codec": "FFV1 lossless (decord bit-exact)", "old_mp4v_target": ORIGIN_SBS_MP4V_OLD,
           "sanity_mad_left_vs_splat_TL_255": mad_spl, "sanity_maxabs_left_vs_splat_TL_255": maxabs_spl, "start_point_file": "start_bf16.pt"}, open(os.path.join(OUT, "meta.json"), "w"), indent=1)

# ---- training ----
g = torch.Generator(device=dev).manual_seed(SEED); rng = torch.Generator().manual_seed(SEED + 1)
csvf = open(os.path.join(OUT, "train_log.csv"), "w", newline=""); wr = csv.writer(csvf)
wr.writerow(["step", "win_start", "sig_idx", "sigma", "loss", "grad_norm_preclip", "step_s", "peak_alloc_GiB"]); csvf.flush()
mparams = [master[n] for n, _ in TRAIN]
t_all = time.time()
for step in range(1, STEPS + 1):
    s, emb, lat, ml, x0, add = WINS[(step - 1) % len(WINS)]
    idx, sigma, t_val = draw_sigma(rng); den = math.sqrt(sigma ** 2 + 1)
    t = torch.tensor([t_val], dtype=torch.float32, device=dev)
    eps = torch.randn(x0.shape, generator=g, device=dev, dtype=torch.float32); x0f = x0.float()
    x_t = ((x0f + eps * sigma) / den).to(dt); target = (eps - sigma * x0f) / den
    torch.cuda.synchronize(); t0 = time.time()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        pred = pipe.unet(torch.cat([x_t, lat, ml], dim=2), t, encoder_hidden_states=emb, added_time_ids=add, return_dict=False)[0]
    loss = (pred.float() - target).pow(2).mean(); loss.backward()
    for n, p in TRAIN:
        master[n].grad = p.grad.detach().float(); p.grad = None
    gn = float(torch.nn.utils.clip_grad_norm_(mparams, 1.0))
    opt.step(); opt.zero_grad(set_to_none=True)
    with torch.no_grad():
        for n, p in TRAIN: p.copy_(master[n].to(dt))
    torch.cuda.synchronize(); st = time.time() - t0; pk = torch.cuda.max_memory_allocated() / 2**30
    wr.writerow([step, s, idx, f"{sigma:.4g}", f"{loss.item():.6f}", f"{gn:.4f}", f"{st:.3f}", f"{pk:.2f}"]); csvf.flush()
    if step == 1: print(f"[mem] peak alloc after step 1: {pk:.2f} GiB, reserved {torch.cuda.max_memory_reserved()/2**30:.2f} GiB, step {st:.2f}s", flush=True)
    if step % 10 == 0 or step == 1: print(f"step {step} win {s} sigma {sigma:.4g} loss {loss.item():.4f} gn {gn:.3f} {st:.2f}s", flush=True)
    if step in SAVE_AT:
        sd = {n: master[n].detach().cpu().clone() for n, _ in TRAIN}
        torch.save(sd, os.path.join(OUT, f"step{step}.pt")); print(f"saved step{step}.pt", flush=True)
    del pred, loss, x_t, target, eps
print(f"DONE {VARIANT} {STEPS} steps in {time.time()-t_all:.1f}s, peak alloc {torch.cuda.max_memory_allocated()/2**30:.2f} GiB", flush=True)
print("MINIFT_DONE", flush=True)
