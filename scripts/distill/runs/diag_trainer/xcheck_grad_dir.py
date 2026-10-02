"""Gradient-direction cross-check for the 25 trainable attn1 tensors at the trainer's operating point (nf=2, 576x1024, bf16).
Configs (same frames, same eps, same sigma):
  A  trainer   : trainer loader (misregistered quadrants), cond x0.18215   == what the control run optimised
  B  reg_x0.18 : registered quadrants (utils/inpainting.py split), cond x0.18215   -> A vs B isolates the quadrant bug
  C  reg_x1.0  : registered, cond x1.0 (deployed scale)                            -> B vs C isolates the cond-scale bug
  D  reg4_x1.0 : registered, cond x1.0, 4-frame window, loss on the same 2 middle frames -> C vs D isolates window length
Reports cos(gA,gB), cos(gB,gC), cos(gC,gD), cos(gA,gD) over the concatenated 25-tensor gradient, per sigma, and per-tensor min.
"""
import os, sys, json, time
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
CLIPS = ["0154", "0011"]; WIN_STARTS = [40, 100]; SIG_IDX = [int(x) for x in os.environ.get("XSIG", "0,10,14").split(",")]   # grid idx: 0=700 10=11.9 13=1.79 14=0.83 16=0.13
PREF = ["down_blocks.0.attentions.0.transformer_blocks.0.attn1.", "down_blocks.0.attentions.1.transformer_blocks.0.attn1.",
        "up_blocks.3.attentions.0.transformer_blocks.0.attn1.", "up_blocks.3.attentions.1.transformer_blocks.0.attn1.",
        "up_blocks.3.attentions.2.transformer_blocks.0.attn1."]
pre = "weights/stable-video-diffusion-img2vid-xt-1-1/"; unet_path = "weights/StereoCrafter/"
image_encoder = CLIPVisionModelWithProjection.from_pretrained(pre, subfolder="image_encoder", variant="fp16", torch_dtype=dt)
vae = AutoencoderKLTemporalDecoder.from_pretrained(pre, subfolder="vae", variant="fp16", torch_dtype=dt)
unet = UNetSpatioTemporalConditionModel.from_pretrained(unet_path, subfolder="unet_diffusers", low_cpu_mem_usage=True, torch_dtype=dt)
pipe = _Pipe.from_pretrained(pre, image_encoder=image_encoder, vae=vae, unet=unet, torch_dtype=dt).to(dev)
pipe.vae.eval(); pipe.image_encoder.eval()
for n, p in pipe.unet.named_parameters(): p.requires_grad_(any(n.startswith(q) for q in PREF) and "origin_attn" not in n)
TRAIN = [(n, p) for n, p in pipe.unet.named_parameters() if p.requires_grad]; print("trainable", len(TRAIN), flush=True)
pipe.unet.enable_gradient_checkpointing(); pipe.unet.train()   # checkpointing only active in train(); SVD has no dropout/batchnorm
sched = EulerDiscreteScheduler.from_config(pipe.scheduler.config); sched.set_timesteps(20, device=dev)

def aligned_batch(clip, s, e):
    vr = VideoReader(f"video_data/train_gt28/{clip}_train.mp4", ctx=cpu(0))
    fr = torch.from_numpy(vr.get_batch(list(range(s, e))).asnumpy()).permute(0, 3, 1, 2).float() / 255.0
    H, W = fr.shape[2] // 2, fr.shape[3] // 2
    TR, BL, BR = fr[:, :, :H, W:], fr[:, :, H:, :W], fr[:, :, H:, W:]
    h, w = H // 128 * 128, W // 128 * 128
    TR, BL, BR = TR[:, :, :h, :w], BL[:, :, :h, :w], BR[:, :, :h, :w]
    top, left = (h - 576) // 2, (w - 1024) // 2
    sl = (slice(None), slice(None), slice(top, top + 576), slice(left, left + 1024))
    return BR[sl], BL[sl].mean(dim=1, keepdim=True), TR[sl]
def trainer_batch(clip, s, e):
    bi = prepare_batches(f"video_data/train_gt28/{clip}_train.mp4", frames_chunk=2, overlap=1, device=torch.device("cpu"), dtype=torch.float32,
                         crop_multiple=64, crop_min_size=(576, 1024), crop_max_size=(576, 1024), random_crop=False, use_prev_target_overlap=False)
    bi._ranges = [(s, e)]; b = next(iter(bi)); return b.cond, b.mask, b.target
@torch.no_grad()
def encode(cond, mask, target, cond_scale):
    cond, mask, target = cond.to(dev, dt), mask.to(dev, dt), target.to(dev, dt); H, W = cond.shape[2], cond.shape[3]
    emb = pipe._encode_image(cond[0:1].float(), device=dev, num_videos_per_prompt=1, do_classifier_free_guidance=False)
    fc = pipe.image_processor.preprocess(cond, height=H, width=W)
    lat = torch.cat([pipe.vae.encode(fc[i:i + 1]).latent_dist.mode() for i in range(fc.shape[0])], 0).unsqueeze(0) * cond_scale
    fm = pipe.mask_processor.preprocess(mask, height=H, width=W); ml = F.interpolate(fm, scale_factor=1 / pipe.vae_scale_factor).unsqueeze(0).to(dt)
    ft = pipe.image_processor.preprocess(target, height=H, width=W)
    x0 = torch.cat([pipe.vae.encode(ft[i:i + 1]).latent_dist.mode() for i in range(ft.shape[0])], 0).unsqueeze(0) * pipe.vae.config.scaling_factor
    return emb, lat.to(dt), ml, x0.to(dt), torch.tensor([[6.0, 127.0, 0.0]], dtype=dt, device=dev)
def grad_at(emb, lat, ml, x0, add, idx, eps, score_frames=None):
    t = sched.timesteps[idx].to(torch.float32).reshape(1); sigma = sched.sigmas[idx].to(dt); den = (sigma ** 2 + 1).sqrt()
    x_t = ((x0 + eps * sigma) / den).to(dt); target = ((eps - sigma * x0) / den)
    for _, p in TRAIN: p.grad = None
    pred = pipe.unet(torch.cat([x_t, lat, ml], dim=2), t, encoder_hidden_states=emb, added_time_ids=add, return_dict=False)[0]
    if score_frames is not None: pred, target = pred[:, score_frames], target[:, score_frames]
    loss = (pred.float() - target.float()).pow(2).mean(); loss.backward()
    g = torch.cat([p.grad.detach().float().flatten() for _, p in TRAIN]); per = {n: p.grad.detach().float().flatten().clone() for n, p in TRAIN}
    return loss.item(), g, per
def cos(a, b): return float(F.cosine_similarity(a, b, dim=0))
OUT = {"pairs": {}, "per_window": []}; t0 = time.time()
acc = {k: [] for k in ["A_B", "B_C", "C_D", "A_D", "A_C"]}; accsig = {}
pmin = {k: [] for k in acc}
for clip in CLIPS:
    for s in WIN_STARTS:
        cfg = {}
        cfg["A"] = encode(*trainer_batch(clip, s, s + 2), pipe.vae.config.scaling_factor)
        reg2 = aligned_batch(clip, s, s + 2)
        cfg["B"] = encode(*reg2, pipe.vae.config.scaling_factor)
        cfg["C"] = encode(*reg2, 1.0)
        cfg["D"] = encode(*aligned_batch(clip, s - 1, s + 3), 1.0)
        g = torch.Generator(device=dev).manual_seed(1000 + s)
        eps2 = torch.randn(cfg["A"][3].shape, generator=g, device=dev, dtype=torch.float32).to(dt)
        eps4 = torch.randn(cfg["D"][3].shape, generator=g, device=dev, dtype=torch.float32).to(dt); eps4[:, 1:3] = eps2
        for idx in SIG_IDX:
            sig = round(float(sched.sigmas[idx]), 3); res = {"clip": clip, "win": s, "sigma": sig, "loss": {}, "cos": {}, "gnorm": {}}
            G = {}; P = {}
            for k in "ABC":
                l, G[k], P[k] = grad_at(*cfg[k], idx, eps2); res["loss"][k] = round(l, 4); res["gnorm"][k] = round(float(G[k].norm()), 3)
            l, G["D"], P["D"] = grad_at(*cfg["D"], idx, eps4, score_frames=slice(1, 3)); res["loss"]["D"] = round(l, 4); res["gnorm"]["D"] = round(float(G["D"].norm()), 3)
            for pair in acc:
                a, b = pair.split("_"); c = cos(G[a], G[b]); res["cos"][pair] = round(c, 3); acc[pair].append(c); accsig.setdefault((pair, sig), []).append(c)
                pmin[pair].append(min(cos(P[a][n], P[b][n]) for n in P[a]))
            OUT["per_window"].append(res); print(json.dumps(res), flush=True)
for pair in acc:
    OUT["pairs"][pair] = {"mean_cos": round(sum(acc[pair]) / len(acc[pair]), 3), "min_per_tensor_cos": round(min(pmin[pair]), 3),
                          "by_sigma": {str(sig): round(sum(v) / len(v), 3) for (p, sig), v in accsig.items() if p == pair}}
OUT["_elapsed_s"] = round(time.time() - t0, 1); OUT["_mem_GiB"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
print("SUMMARY", json.dumps(OUT["pairs"]), OUT["_elapsed_s"], "s", OUT["_mem_GiB"], "GiB", flush=True)
json.dump(OUT, open("scripts/distill/runs/diag_trainer/xcheck_grad_dir" + os.environ.get("XTAG", "") + ".json", "w"), indent=1); print("GRAD_DONE", flush=True)
