"""Origin's training loss on the trainer's (misregistered) batches vs correctly registered batches.
Also: vertical latent shift that minimises origin's x0-prediction error vs the target latent.
Short single-GPU job (<10 min). Nothing is written except a JSON under scripts/distill/runs/diag_trainer/."""
import os, sys, json, math, time
REPO = "/home/kawa/master_project/StereoCrafter"; os.chdir(REPO); sys.path.insert(0, REPO)
os.environ.setdefault("MAMBA_SELF_ATTN_INCLUDE", "__nomatch__")
import torch, torch.nn.functional as F
from decord import VideoReader, cpu
from transformers import CLIPVisionModelWithProjection
from diffusers.models.unets.unet_spatio_temporal_condition import UNetSpatioTemporalConditionModel
from diffusers.models.autoencoders.autoencoder_kl_temporal_decoder import AutoencoderKLTemporalDecoder
from diffusers import EulerDiscreteScheduler
from pipelines.stereo_video_inpainting import StableVideoDiffusionInpaintingPipeline as _Pipe
from utils.training_batches import prepare_batches

dev = torch.device("cuda:0"); dt = torch.bfloat16
CLIPS = sys.argv[1:] or ["0204", "0042"]
WIN_STARTS = [20, 60, 100, 130]
SIG_IDX = [int(x) for x in os.environ.get("LENS3_SIG", "0,4,8,12,16,19").split(",")]
SHIFT_IDX = set(int(x) for x in os.environ.get("LENS3_SHIFT", "12,16").split(","))
pre = "weights/stable-video-diffusion-img2vid-xt-1-1/"; unet_path = "weights/StereoCrafter/"
image_encoder = CLIPVisionModelWithProjection.from_pretrained(pre, subfolder="image_encoder", variant="fp16", torch_dtype=dt)
vae = AutoencoderKLTemporalDecoder.from_pretrained(pre, subfolder="vae", variant="fp16", torch_dtype=dt)
unet = UNetSpatioTemporalConditionModel.from_pretrained(unet_path, subfolder="unet_diffusers", low_cpu_mem_usage=True, torch_dtype=dt)
pipe = _Pipe.from_pretrained(pre, image_encoder=image_encoder, vae=vae, unet=unet, torch_dtype=dt).to(dev)
CK = os.environ.get("LENS3_CKPT")
if CK:
    raw = torch.load(CK, map_location="cpu", weights_only=False)
    sd = None
    if isinstance(raw, dict):
        for key in ("model", "unet", "state_dict"):
            if key in raw and isinstance(raw[key], dict): sd = raw[key]; break
        if sd is None and all(isinstance(v, torch.Tensor) for v in raw.values()): sd = raw
    pipe.unet.to(torch.float32); missing, unexpected = pipe.unet.load_state_dict(sd, strict=False); pipe.unet.to(dt)
    print("LOADED", CK, "tensors", len(sd), "missing", len(missing), "unexpected", len(unexpected), flush=True)
pipe.unet.eval(); pipe.vae.eval(); pipe.image_encoder.eval()
sched = EulerDiscreteScheduler.from_config(pipe.scheduler.config); sched.set_timesteps(20, device=dev)
print("sigmas", [round(float(s), 2) for s in sched.sigmas[:20]], "pred", sched.config.prediction_type)

def aligned_batch(clip, s, e):
    """Registered loader: split at the TRUE half, crop each quadrant to //128 (like utils/inpainting.py:145-159), centre-crop 576x1024."""
    vr = VideoReader(f"video_data/train/{clip}_train.mp4", ctx=cpu(0))
    fr = torch.from_numpy(vr.get_batch(list(range(s, e))).asnumpy()).permute(0, 3, 1, 2).float() / 255.0
    H, W = fr.shape[2] // 2, fr.shape[3] // 2
    TR, BL, BR = fr[:, :, :H, W:], fr[:, :, H:, :W], fr[:, :, H:, W:]
    h, w = H // 128 * 128, W // 128 * 128
    TR, BL, BR = TR[:, :, :h, :w], BL[:, :, :h, :w], BR[:, :, :h, :w]
    top, left = (h - 576) // 2, (w - 1024) // 2
    sl = (slice(None), slice(None), slice(top, top + 576), slice(left, left + 1024))
    return BR[sl], BL[sl].mean(dim=1, keepdim=True), TR[sl]

def trainer_batch(clip, s, e):
    bi = prepare_batches(f"video_data/train/{clip}_train.mp4", frames_chunk=2, overlap=1, device=torch.device("cpu"), dtype=torch.float32,
                         crop_multiple=64, crop_min_size=(576, 1024), crop_max_size=(576, 1024), random_crop=False, use_prev_target_overlap=False)
    bi._ranges = [(s, e)]
    b = next(iter(bi)); return b.cond, b.mask, b.target

@torch.no_grad()
def encode(cond, mask, target):
    cond, mask, target = cond.to(dev, dt), mask.to(dev, dt), target.to(dev, dt)
    H, W = cond.shape[2], cond.shape[3]
    emb = pipe._encode_image(cond[0:1].float(), device=dev, num_videos_per_prompt=1, do_classifier_free_guidance=False)
    fc = pipe.image_processor.preprocess(cond, height=H, width=W)
    lat = torch.cat([pipe.vae.encode(fc[i:i + 1]).latent_dist.mode() for i in range(fc.shape[0])], 0).unsqueeze(0) * pipe.vae.config.scaling_factor
    fm = pipe.mask_processor.preprocess(mask, height=H, width=W)
    ml = F.interpolate(fm, scale_factor=1 / pipe.vae_scale_factor).unsqueeze(0).to(dt)
    ft = pipe.image_processor.preprocess(target, height=H, width=W)
    x0 = torch.cat([pipe.vae.encode(ft[i:i + 1]).latent_dist.mode() for i in range(ft.shape[0])], 0).unsqueeze(0) * pipe.vae.config.scaling_factor
    add = torch.tensor([[6.0, 127.0, 0.0]], dtype=dt, device=dev)
    return emb, lat.to(dt), ml, x0.to(dt), add

@torch.no_grad()
def loss_at(emb, lat, ml, x0, add, idx, eps):
    t = sched.timesteps[idx].to(torch.float32).reshape(1); sigma = sched.sigmas[idx].to(dt)
    den = (sigma ** 2 + 1).sqrt()
    x_t = ((x0 + eps * sigma) / den).to(dt)
    target = ((eps - sigma * x0) / den)
    pred = pipe.unet(torch.cat([x_t, lat, ml], dim=2), t, encoder_hidden_states=emb, added_time_ids=add, return_dict=False)[0]
    mse = (pred.float() - target.float()).pow(2).mean().item()
    x0_pred = (x_t.float() - sigma.float() * pred.float()) / den.float()
    return mse, x0_pred

def shift_err(x0_pred, x0, maxs=8):
    """MSE between x0_pred and x0 shifted vertically by dy latent rows (positive = x0 content moved down)."""
    out = {}
    Hl = x0.shape[3]
    for dy in range(-maxs, maxs + 1):
        if dy >= 0: a, b = x0_pred[..., dy:, :], x0.float()[..., :Hl - dy, :]
        else: a, b = x0_pred[..., :Hl + dy, :], x0.float()[..., -dy:, :]
        out[dy] = round((a - b).pow(2).mean().item(), 4)
    best = min(out, key=out.get); return out, best

OUT = {}
t0 = time.time()
for clip in CLIPS:
    R = {"per_window": [], "mean_loss": {}, "shift_best": {}}
    sums = {"trainer": [], "aligned": []}; bests = {"trainer": [], "aligned": []}
    for s in WIN_STARTS:
        e = s + 2
        batches = {"trainer": trainer_batch(clip, s, e), "aligned": aligned_batch(clip, s, e)}
        # sanity: targets identical between the two loaders, conds differ
        tdiff = (batches["trainer"][2] - batches["aligned"][2]).abs().mean().item() * 255
        cdiff = (batches["trainer"][0] - batches["aligned"][0]).abs().mean().item() * 255
        enc = {k: encode(*v) for k, v in batches.items()}
        g = torch.Generator(device=dev).manual_seed(1000 + s)
        rec = {"win": [s, e], "target_MAD_trainer_vs_aligned": round(tdiff, 3), "cond_MAD_trainer_vs_aligned": round(cdiff, 3), "loss": {}, "shift": {}}
        for idx in SIG_IDX:
            eps = torch.randn(enc["aligned"][3].shape, generator=g, device=dev, dtype=torch.float32).to(dt)
            for k in ("trainer", "aligned"):
                mse, x0p = loss_at(*enc[k], idx, eps)
                rec["loss"].setdefault(k, {})[idx] = round(mse, 4); sums[k].append(mse)
                if idx in SHIFT_IDX:
                    errs, best = shift_err(x0p, enc[k][3]); rec["shift"].setdefault(k, {})[idx] = {"best_dy": best, "err_at_0": errs[0], "err_at_best": errs[best], "err_at_-7": errs[-7], "err_at_-3": errs[-3], "err_at_+3": errs[3], "err_at_+7": errs[7]}
                    bests[k].append(best)
        R["per_window"].append(rec); print(json.dumps(rec), flush=True)
    for k in sums: R["mean_loss"][k] = round(sum(sums[k]) / len(sums[k]), 4); R["shift_best"][k] = bests[k]
    OUT[clip] = R
    print(clip, "MEAN LOSS trainer=%.4f aligned=%.4f" % (R["mean_loss"]["trainer"], R["mean_loss"]["aligned"]), "shift bests", R["shift_best"], flush=True)
OUT["_elapsed_s"] = round(time.time() - t0, 1)
json.dump(OUT, open("scripts/distill/runs/diag_trainer/lens3_loss_alignment" + os.environ.get("LENS3_TAG", "") + ".json", "w"), indent=1)
print("WROTE scripts/distill/runs/diag_trainer/lens3_loss_alignment.json", OUT["_elapsed_s"])
