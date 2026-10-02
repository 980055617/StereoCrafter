"""LENS 2(b): the trainer's exact diffusion loss on a FIXED set of held-out windows.

Replicates the euler / v-prediction branch of `compute_batch_loss` in inpainting_train.py
(lines 933-1160 on branch attn1_only_version) without importing the trainer module:

  image_embeddings = pipeline._encode_image(cond[0:1], ...)                      (train :957-959, infer :468)
  cond latents     = vae.encode(preprocess(cond)).latent_dist.mode()             (train :969-977)
                     * vae.config.scaling_factor                                  (train :979)   <-- trainer only
                     (inference feeds them UNscaled: pipelines/stereo_video_inpainting.py:183-212 + :481-482)
  mask latents     = interpolate(mask_processor.preprocess(mask), 1/8)           (train :983-987, infer :214-232)
  x0               = vae.encode(preprocess(target)).mode() * scaling_factor       (train :999-1011)
  sigma            = scheduler.sigmas.to(bf16)[idx]; denom = sqrt(sigma^2 + 1)    (train :1053-1057)
  x_t              = (x0 + eps*sigma) / denom  (bf16)                             (train :1058-1059)
  target           = (eps - sigma*x0) / denom                                     (train :1061-1062)
  t                = scheduler.timesteps.float()[idx]  (= 0.25*log(sigma))       (train :1052)
  loss             = mean((unet(cat[x_t, cond, mask], t).float() - target.float())^2)   (train :1112-1120)

For every (window, sigma) the SAME eps is used for all models and both cond scalings, so the
comparison origin vs epoch-1 vs epoch-2 is paired.

Outputs (never overwrites an existing file): <out>.json (all records) and <out>.txt (summary tables).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict

REPO = "/home/kawa/master_project/StereoCrafter"
os.chdir(REPO)
sys.path.insert(0, REPO)

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKLTemporalDecoder, UNetSpatioTemporalConditionModel
from diffusers.schedulers import EulerDiscreteScheduler
from transformers import CLIPVisionModelWithProjection

from pipelines.stereo_video_inpainting import StableVideoDiffusionInpaintingPipeline
from utils.training_batches import prepare_batches

PRE = "weights/stable-video-diffusion-img2vid-xt-1-1/"
UNET_PATH = "weights/StereoCrafter/"
RUN = "weights/GTfinetune_v2_originattn_control/MambaCrafter_20260925_143200"
CKPTS = {
    "e1": f"{RUN}/train_state_epoch000001.pt",
    "e2": f"{RUN}/train_state_epoch000002.pt",
}
KEEP = [
    "down_blocks.0.attentions.0.transformer_blocks.0.attn1.",
    "down_blocks.0.attentions.1.transformer_blocks.0.attn1.",
    "up_blocks.3.attentions.0.transformer_blocks.0.attn1.",
    "up_blocks.3.attentions.1.transformer_blocks.0.attn1.",
    "up_blocks.3.attentions.2.transformer_blocks.0.attn1.",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--clips", default="0042,0170,0301")
    p.add_argument("--windows-per-clip", type=int, default=8)
    p.add_argument("--frames-chunk", type=int, default=2, help="stage-3 trainer value = 2; inference = 14")
    p.add_argument("--overlap", type=int, default=1, help="stage-3 trainer value = 1; inference = 3")
    p.add_argument("--stage-h", type=int, default=576)
    p.add_argument("--stage-w", type=int, default=1024)
    p.add_argument("--sigma-sets", default="inference,grid", help="inference = 8-step Euler grid; grid = 20-step train grid")
    p.add_argument("--cond-scales", default="trainer,inference",
                   help="trainer = cond latents * scaling_factor (train.py:979); inference = unscaled (pipeline)")
    p.add_argument("--models", default="origin,e1,e2")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", required=True, help="output prefix (json + txt); refuses to overwrite")
    p.add_argument("--max-sigmas", type=int, default=0, help="smoke: only the first N sigmas of each set")
    return p.parse_args()


def load_slots(path: str, keys: list[str], device) -> dict[str, torch.Tensor]:
    raw = torch.load(path, map_location="cpu", weights_only=False)
    sd = raw["model"]
    out = {k: sd[k].detach().clone().to(device) for k in keys}
    del raw, sd
    return out


@torch.no_grad()
def apply_slots(params: dict[str, torch.nn.Parameter], slots: dict[str, torch.Tensor]) -> None:
    for k, v in slots.items():
        params[k].copy_(v)


def main():
    args = parse_args()
    out_json = args.out + ".json"
    out_txt = args.out + ".txt"
    for f in (out_json, out_txt):
        if os.path.exists(f):
            raise SystemExit(f"refusing to overwrite {f}")
    device = torch.device(args.device)
    dtype = torch.bfloat16
    torch.manual_seed(args.seed)

    # --- pipeline exactly as the trainer / origin inference build it (bf16 everywhere) ---
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(PRE, subfolder="image_encoder", variant="fp16", torch_dtype=dtype)
    vae = AutoencoderKLTemporalDecoder.from_pretrained(PRE, subfolder="vae", variant="fp16", torch_dtype=dtype)
    unet = UNetSpatioTemporalConditionModel.from_pretrained(UNET_PATH, subfolder="unet_diffusers", low_cpu_mem_usage=True, torch_dtype=dtype)
    pipeline = StableVideoDiffusionInpaintingPipeline.from_pretrained(PRE, image_encoder=image_encoder, vae=vae, unet=unet, torch_dtype=dtype)
    pipeline = pipeline.to(device)
    pipeline.unet.eval(); pipeline.vae.eval(); pipeline.image_encoder.eval()
    scaling_factor = float(pipeline.vae.config.scaling_factor)

    # --- sigma sets ---
    sigma_sets = {}
    for name in args.sigma_sets.split(","):
        sched = EulerDiscreteScheduler.from_config(pipeline.scheduler.config)
        n = 8 if name == "inference" else 20
        sched.set_timesteps(n, device=device)
        idxs = list(range(n))
        if args.max_sigmas > 0:
            idxs = idxs[: args.max_sigmas]
        sigma_sets[name] = (sched, idxs)
        print(f"[sigmas] {name}: {[round(float(s), 4) for s in sched.sigmas[:n].tolist()]}", flush=True)

    # --- model slot sets ---
    params = dict(pipeline.unet.named_parameters())
    keys = [k for k in params if any(k.startswith(p) for p in KEEP)]
    assert len(keys) == 25, len(keys)
    slot_sets = {}
    for m in args.models.split(","):
        if m == "origin":
            slot_sets[m] = {k: params[k].detach().clone() for k in keys}
        else:
            slot_sets[m] = load_slots(CKPTS[m], keys, device)
            rel = math.sqrt(sum((slot_sets[m][k].float() - slot_sets["origin"][k].float()).pow(2).sum().item() for k in keys)
                            / sum(slot_sets["origin"][k].float().pow(2).sum().item() for k in keys))
            print(f"[slots] {m}: 25 tensors loaded, aggregate rel |delta| vs origin = {rel:.5f}", flush=True)
    cond_scales = {"trainer": scaling_factor, "inference": 1.0}
    cond_scales = {k: cond_scales[k] for k in args.cond_scales.split(",")}

    add_time_ids = torch.tensor([[6.0, 127.0, 0.0]], dtype=dtype, device=device)  # fps 7-1, motion 127, noise_aug 0 (train :989-995)

    # --- fixed windows ---
    windows = []
    for clip in args.clips.split(","):
        path = f"video_data/train/{clip}_train.mp4"
        tb = prepare_batches(
            path, frames_chunk=args.frames_chunk, overlap=args.overlap, device=device, dtype=dtype,
            crop_multiple=64, crop_min_size=(args.stage_h, args.stage_w), crop_max_size=(args.stage_h, args.stage_w),
            random_crop=False, use_prev_target_overlap=False,
        )
        ranges = list(tb._ranges)
        sel = sorted(set(int(round(x)) for x in np.linspace(0, len(ranges) - 1, args.windows_per_clip)))
        tb._ranges = [ranges[i] for i in sel]
        for (start, end), batch in zip(tb._ranges, tb):
            windows.append((clip, start, end, batch))
        print(f"[windows] {clip}: {len(ranges)} ranges, using {[(ranges[i][0], ranges[i][1]) for i in sel]}", flush=True)

    # --- precompute per-window conditioning (train :957-1011) ---
    pre = []
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype):
        for clip, start, end, b in windows:
            H, W = b.cond.shape[2], b.cond.shape[3]
            emb = pipeline._encode_image(b.cond[0:1], device, 1, False)
            fc = pipeline.image_processor.preprocess(b.cond, height=H, width=W)
            cond_lat = torch.cat([pipeline.vae.encode(fc[i:i + 1]).latent_dist.mode() for i in range(fc.shape[0])], 0).unsqueeze(0)
            cond_lat = cond_lat.to(device=device, dtype=emb.dtype)  # UNscaled (what inference feeds)
            fm = pipeline.mask_processor.preprocess(b.mask, height=H, width=W)
            mask_lat = F.interpolate(fm, scale_factor=1 / pipeline.vae_scale_factor).unsqueeze(0).to(device=device, dtype=emb.dtype)
            ft = pipeline.image_processor.preprocess(b.target, height=H, width=W)
            x0 = torch.cat([pipeline.vae.encode(ft[i:i + 1]).latent_dist.mode() for i in range(ft.shape[0])], 0).unsqueeze(0).to(emb.dtype)
            x0 = x0 * scaling_factor  # (train :1011)
            pre.append(dict(clip=clip, start=start, end=end, emb=emb, cond_lat=cond_lat, mask_lat=mask_lat, x0=x0,
                            mask_frac=float(mask_lat.float().mean().item())))
    del windows
    print(f"[pre] {len(pre)} windows; latent shape {tuple(pre[0]['x0'].shape)}; cond_lat rms unscaled={pre[0]['cond_lat'].float().pow(2).mean().sqrt():.3f} "
          f"x0 rms={pre[0]['x0'].float().pow(2).mean().sqrt():.3f}", flush=True)

    records = []
    n_fwd = 0
    t0 = time.time()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype):
        for wi, w in enumerate(pre):
            x0 = w["x0"]
            for sname, (sched, idxs) in sigma_sets.items():
                for si in idxs:
                    g = torch.Generator(device=device).manual_seed(args.seed * 1000003 + wi * 1009 + (0 if sname == "inference" else 500) + si)
                    eps = torch.randn(x0.shape, generator=g, device=device, dtype=x0.dtype)  # randn_like(x0) in bf16 (train :1032)
                    sigma_value = sched.sigmas.to(device=device, dtype=x0.dtype)[si]           # bf16 sigma (train :1053)
                    sigma = sigma_value.reshape(1, 1, 1, 1, 1)
                    denom = (sigma.pow(2) + 1.0).sqrt()
                    x_t = ((x0 + eps * sigma) / denom).to(dtype)                                # (train :1058-1059)
                    target = (eps - sigma * x0) / denom                                          # (train :1062)
                    t = sched.timesteps.to(device=device, dtype=torch.float32)[si:si + 1]      # (train :1052)
                    mask_w = w["mask_lat"].float().clamp(0, 1).expand_as(x0)
                    for cname, cscale in cond_scales.items():
                        cond = w["cond_lat"] * cscale if cscale != 1.0 else w["cond_lat"]       # (train :979 when cscale=scaling_factor)
                        inp = torch.cat([x_t, cond, w["mask_lat"]], dim=2)
                        for mname, slots in slot_sets.items():
                            apply_slots(params, slots)
                            pred = pipeline.unet(inp, t, encoder_hidden_states=w["emb"], added_time_ids=add_time_ids, return_dict=False)[0]
                            n_fwd += 1
                            err2 = (pred.float() - target.float()).pow(2)
                            loss = err2.mean().item()                                          # (train :1112-1113, noise_mask_loss_weight = 0)
                            m_sum = mask_w.sum().item()
                            loss_masked = (err2 * mask_w).sum().item() / max(m_sum, 1.0)
                            loss_unmasked = (err2 * (1 - mask_w)).sum().item() / max((1 - mask_w).sum().item(), 1.0)
                            x0_pred = (x_t.float() - sigma.float() * pred.float()) / denom.float()  # (train :1142-1144)
                            x0_mse = (x0_pred - x0.float()).pow(2).mean().item()
                            records.append(dict(clip=w["clip"], start=w["start"], end=w["end"], window=wi, sigma_set=sname, sigma_idx=si,
                                                sigma=float(sigma_value.float().item()), timestep=float(t.item()), cond_scale=cname,
                                                model=mname, loss=loss, loss_masked=loss_masked, loss_unmasked=loss_unmasked,
                                                x0_mse=x0_mse, mask_frac=w["mask_frac"]))
            el = time.time() - t0
            print(f"[progress] window {wi + 1}/{len(pre)} ({w['clip']} {w['start']}-{w['end']}) forwards={n_fwd} elapsed={el / 60:.1f} min "
                  f"eta={(el / (wi + 1)) * (len(pre) - wi - 1) / 60:.1f} min", flush=True)
    apply_slots(params, slot_sets["origin"])

    # --- summaries ---
    lines = []
    def emit(s=""):
        lines.append(s); print(s, flush=True)
    emit(f"# val_loss  clips={args.clips} windows={len(pre)} frames_chunk={args.frames_chunk} overlap={args.overlap} "
         f"crop={args.stage_h}x{args.stage_w} seed={args.seed} forwards={n_fwd} elapsed={(time.time() - t0) / 60:.1f} min")
    emit(f"# scaling_factor={scaling_factor}; cond_scale 'trainer' = *{scaling_factor} (inpainting_train.py:979), 'inference' = unscaled (pipeline)")
    models = list(slot_sets)
    for metric in ("loss", "x0_mse", "loss_masked", "loss_unmasked"):
        for cname in cond_scales:
            for sname, (sched, idxs) in sigma_sets.items():
                emit(f"\n== {metric} | cond_scale={cname} | sigma_set={sname} | mean over {len(pre)} windows; d = model - origin; "
                     f"wins = #windows where model < origin ==")
                emit(f"{'idx':>3} {'sigma':>9} " + " ".join(f"{m:>9}" for m in models) + " " + " ".join(f"d({m}):>10" for m in models[1:]) + " " + " ".join(f"{'wins(' + m + ')':>10}" for m in models[1:]))
                tot = defaultdict(list)
                for si in idxs:
                    per = {}
                    for m in models:
                        vals = {r["window"]: r[metric] for r in records if r["sigma_set"] == sname and r["sigma_idx"] == si and r["cond_scale"] == cname and r["model"] == m}
                        per[m] = vals
                    ws = sorted(per[models[0]])
                    means = {m: float(np.mean([per[m][x] for x in ws])) for m in models}
                    for m in models:
                        tot[m].append(means[m])
                    sig = float(sched.sigmas[si])
                    row = f"{si:>3} {sig:>9.4f} " + " ".join(f"{means[m]:>9.4f}" for m in models)
                    row += " " + " ".join(f"{means[m] - means[models[0]]:>+10.4f}" for m in models[1:])
                    row += " " + " ".join(f"{sum(per[m][x] < per[models[0]][x] for x in ws):>7d}/{len(ws):<2d}" for m in models[1:])
                    emit(row)
                emit(f"{'all':>3} {'':>9} " + " ".join(f"{np.mean(tot[m]):>9.4f}" for m in models) + " " + " ".join(f"{np.mean(tot[m]) - np.mean(tot[models[0]]):>+10.4f}" for m in models[1:]))
    # per-clip totals (loss, both scales, both sigma sets)
    emit("\n== per-clip mean loss (all sigmas of the set) ==")
    for cname in cond_scales:
        for sname in sigma_sets:
            for clip in args.clips.split(","):
                vals = {m: float(np.mean([r["loss"] for r in records if r["clip"] == clip and r["sigma_set"] == sname and r["cond_scale"] == cname and r["model"] == m])) for m in models}
                emit(f"cond={cname:9s} set={sname:9s} clip={clip}: " + " ".join(f"{m}={vals[m]:.4f}" for m in models) + "  " + " ".join(f"d({m})={vals[m] - vals[models[0]]:+.4f}" for m in models[1:]))
    with open(out_json, "w") as f:
        json.dump(dict(args=vars(args), scaling_factor=scaling_factor, n_forwards=n_fwd, records=records), f)
    with open(out_txt, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[done] wrote {out_json} and {out_txt}", flush=True)


if __name__ == "__main__":
    main()
