"""Single-step v-MSE / x0-MSE probe at the 8 inference sigmas: origin vs fine-tuned checkpoints,
2-frame windows (trainer regime) vs 14-frame windows (inference regime). No training, no video writing."""
import sys, os, json, math, time, torch
import torch.nn.functional as F
sys.path.insert(0, os.getcwd())
from transformers import CLIPVisionModelWithProjection
from diffusers import AutoencoderKLTemporalDecoder, UNetSpatioTemporalConditionModel
from pipelines.stereo_video_inpainting import StableVideoDiffusionInpaintingPipeline
from utils.training_batches import _StreamingVideo

PRE = "weights/stable-video-diffusion-img2vid-xt-1-1/"; UNET = "weights/StereoCrafter/"
RUN = "weights/GTfinetune_v2_originattn_control/MambaCrafter_20260925_143200/"
clips = sys.argv[1].split(",")          # e.g. test:0042,test:0052,train:0011
out_json = sys.argv[2]
models = sys.argv[3].split(",") if len(sys.argv) > 3 else ["origin", "e1", "e2", "origin_bf16"]
dev = torch.device("cuda"); fp16 = torch.float16
NF = 14

image_encoder = CLIPVisionModelWithProjection.from_pretrained(PRE, subfolder="image_encoder", variant="fp16", torch_dtype=fp16)
vae = AutoencoderKLTemporalDecoder.from_pretrained(PRE, subfolder="vae", variant="fp16", torch_dtype=fp16)
unet = UNetSpatioTemporalConditionModel.from_pretrained(UNET, subfolder="unet_diffusers", low_cpu_mem_usage=True, torch_dtype=fp16)
for m in (image_encoder, vae, unet): m.requires_grad_(False)
pipe = StableVideoDiffusionInpaintingPipeline.from_pretrained(PRE, image_encoder=image_encoder, vae=vae, unet=unet, torch_dtype=fp16).to(dev)
pipe.scheduler.set_timesteps(8, device=dev)
sigmas = pipe.scheduler.sigmas[:8].float().tolist(); tsteps = pipe.scheduler.timesteps
print("inference sigmas", [round(s, 3) for s in sigmas], flush=True)
orig_sd = {k: v.detach().cpu().clone() for k, v in pipe.unet.state_dict().items()}

def set_model(name):
    pipe.unet.to(dtype=torch.float32); pipe.unet.load_state_dict(orig_sd, strict=True); pipe.unet.to(dtype=fp16)
    if name == "origin":
        return
    if name == "origin_randpert":
        # control: perturb the same 25 tensors by Gaussian noise with the SAME per-tensor norm as the epoch-1 update
        PREF = ["down_blocks.0.attentions.0.transformer_blocks.0.attn1.", "down_blocks.0.attentions.1.transformer_blocks.0.attn1.",
                "up_blocks.3.attentions.0.transformer_blocks.0.attn1.", "up_blocks.3.attentions.1.transformer_blocks.0.attn1.",
                "up_blocks.3.attentions.2.transformer_blocks.0.attn1."]
        s1 = torch.load(RUN + "train_state_epoch000001.pt", map_location="cpu", mmap=True, weights_only=False)["model"]
        g = torch.Generator(device="cpu").manual_seed(7); tot = 0
        with torch.no_grad():
            for k, p in pipe.unet.named_parameters():
                if any(k.startswith(pfx) for pfx in PREF):
                    d1 = s1[k].float() - orig_sd[k].float().cpu().to(torch.bfloat16).float()
                    n = torch.randn(p.shape, generator=g); n = n / n.norm() * d1.norm()
                    p.data = (p.data.float() + n.to(dev)).to(fp16); tot += 1
        print(f"randpert applied to {tot} tensors", flush=True)
        return
    if name == "origin_bf16":
        with torch.no_grad():
            for p in pipe.unet.parameters(): p.data = p.data.to(torch.bfloat16).to(fp16)
        return
    path = {"e1": RUN + "train_state_epoch000001.pt", "e2": RUN + "train_state_epoch000002.pt"}[name]
    sd = torch.load(path, map_location="cpu", mmap=True, weights_only=False)["model"]
    pipe.unet.to(dtype=torch.float32); missing, unexpected = pipe.unet.load_state_dict(sd, strict=False); pipe.unet.to(dtype=fp16)
    print(f"loaded {name}: missing={len(missing)} unexpected={len(unexpected)}", flush=True)

@torch.no_grad()
def encode(frames, H, W):
    x = pipe.image_processor.preprocess(frames, height=H, width=W)
    pipe.vae.to(device=dev, dtype=torch.float32)
    lat = torch.cat([pipe.vae.encode(x[i:i + 1].to(dev, torch.float32)).latent_dist.mode() for i in range(0, len(x))])
    return (lat * pipe.vae.config.scaling_factor).float()

@torch.no_grad()
def prep_clip(kind, cid):
    path = f"video_data/train/{cid}_train.mp4" if kind == "test" else f"video_data/train_gt28/{cid}_train.mp4"
    sv = _StreamingVideo(path); warped, mask, right = sv.load_chunk(0, NF)
    # fixed center crop 576x1024, as the trainer's stage crop (training_batches.py center crop) and the eval's target_height/width
    ch, cw = 576, 1024; top = (warped.shape[2] - ch) // 2; left = (warped.shape[3] - cw) // 2
    warped, mask, right = (x[:, :, top:top + ch, left:left + cw].contiguous() for x in (warped, mask, right))
    H, W = warped.shape[2], warped.shape[3]
    frame_lat = encode(warped, H, W).unsqueeze(0)                 # [1,F,4,h,w]
    x0 = encode(right, H, W).unsqueeze(0)
    m = pipe.mask_processor.preprocess(mask, height=H, width=W)
    mask_lat = F.interpolate(m, scale_factor=1 / pipe.vae_scale_factor).unsqueeze(0).to(dev).float()
    embs = [pipe._encode_image(warped[j:j + 1].to(dev, fp16), dev, 1, False) for j in range(NF)]  # CLIP of each frame as window-first
    g = torch.Generator(device="cpu").manual_seed(1000 + int(cid))
    eps = torch.randn(x0.shape, generator=g).to(dev)
    return dict(H=H, W=W, frame_lat=frame_lat, x0=x0, mask_lat=mask_lat, embs=embs, eps=eps, mask_frac=(mask_lat > 0.5).float().mean().item())

add_ids = torch.tensor([[6.0, 127.0, 0.0]], dtype=fp16, device=dev)

@torch.no_grad()
def step_metrics(sl, d, sigma, t, emb):
    x0 = d["x0"][:, sl]; eps = d["eps"][:, sl]; fl = d["frame_lat"][:, sl]; ml = d["mask_lat"][:, sl]
    den = math.sqrt(sigma ** 2 + 1)
    xt_raw = x0 + eps * sigma
    xt = (xt_raw / den).to(fp16)
    target = (eps - sigma * x0) / den
    inp = torch.cat([xt, fl.to(fp16), ml.to(fp16)], dim=2)
    v = pipe.unet(inp, t, encoder_hidden_states=emb, added_time_ids=add_ids, return_dict=False)[0].float()
    x0p = v * (-sigma / den) + xt_raw / (sigma ** 2 + 1)
    mm = (ml > 0.5).float().expand_as(x0)
    return dict(vmse=(v - target).pow(2).mean().item(),
                x0mse=(x0p - x0).pow(2).mean().item(),
                x0mse_mask=(((x0p - x0).pow(2) * mm).sum() / mm.sum().clamp(min=1)).item(),
                x0mse_unmask=(((x0p - x0).pow(2) * (1 - mm)).sum() / (1 - mm).sum().clamp(min=1)).item(),
                vrms=v.pow(2).mean().sqrt().item(), trms=target.pow(2).mean().sqrt().item())

data = {c: prep_clip(*c.split(":")) for c in clips}
pipe.vae.to("cpu"); torch.cuda.empty_cache()
print("after prep: max_mem_alloc GiB", round(torch.cuda.max_memory_allocated() / 2**30, 2), flush=True)
for c in clips: print(c, "tile", data[c]["H"], data[c]["W"], "mask_frac", round(data[c]["mask_frac"], 3), flush=True)
results = []
for name in models:
    set_model(name); t0 = time.time()
    for c in clips:
        d = data[c]
        for i, sigma in enumerate(sigmas):
            t = tsteps[i]
            r14 = step_metrics(slice(0, NF), d, sigma, t, d["embs"][0])
            acc = {}
            for j in range(0, NF, 2):
                r = step_metrics(slice(j, j + 2), d, sigma, t, d["embs"][j])
                for k, v in r.items(): acc[k] = acc.get(k, 0.0) + v / (NF // 2)
            for mode, r in (("14f", r14), ("2f", acc)):
                results.append(dict(model=name, clip=c, mode=mode, sigma=sigma, **r))
                print(f"{name:12s} {c:11s} {mode:3s} sigma={sigma:8.3f} vmse={r['vmse']:.4f} x0mse={r['x0mse']:.4f} x0mse_mask={r['x0mse_mask']:.4f} x0mse_unmask={r['x0mse_unmask']:.4f} vrms={r['vrms']:.3f} trms={r['trms']:.3f}", flush=True)
    print(f"model {name} done in {time.time() - t0:.0f}s", flush=True)
    json.dump(results, open(out_json, "w"))
print("max_mem_alloc GiB", round(torch.cuda.max_memory_allocated() / 2**30, 2)); print("PROBE_DONE", flush=True)
