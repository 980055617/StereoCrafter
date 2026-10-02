"""Peak-memory / step-time probe for a deployment-faithful training step (registered quadrants, cond x1.0, 576x1024, bf16,
gradient checkpointing, ff_chunk_size 1 like stage 3) at window lengths 8 and 14, 25 attn1 tensors trainable.
usage: [FF=0|1] [UP3=0|1] CUDA_VISIBLE_DEVICES=1 python xcheck_mem_window.py 8 14   (FF=0 disables ff chunking; UP3=1 trains only the 15 up_blocks.3 tensors, down/mid under no_grad)"""
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
from utils.training_pipeline import configure_unet_memory_features
dev = torch.device("cuda:0"); dt = torch.bfloat16
PREF = ["down_blocks.0.attentions.0.transformer_blocks.0.attn1.", "down_blocks.0.attentions.1.transformer_blocks.0.attn1.",
        "up_blocks.3.attentions.0.transformer_blocks.0.attn1.", "up_blocks.3.attentions.1.transformer_blocks.0.attn1.",
        "up_blocks.3.attentions.2.transformer_blocks.0.attn1."]
pre = "weights/stable-video-diffusion-img2vid-xt-1-1/"; unet_path = "weights/StereoCrafter/"
image_encoder = CLIPVisionModelWithProjection.from_pretrained(pre, subfolder="image_encoder", variant="fp16", torch_dtype=dt)
vae = AutoencoderKLTemporalDecoder.from_pretrained(pre, subfolder="vae", variant="fp16", torch_dtype=dt)
unet = UNetSpatioTemporalConditionModel.from_pretrained(unet_path, subfolder="unet_diffusers", low_cpu_mem_usage=True, torch_dtype=dt)
pipe = _Pipe.from_pretrained(pre, image_encoder=image_encoder, vae=vae, unet=unet, torch_dtype=dt).to(dev)
pipe.vae.eval(); pipe.image_encoder.eval()
UP3 = os.environ.get("UP3", "0") == "1"; FF = int(os.environ.get("FF", "1"))
for n, p in pipe.unet.named_parameters(): p.requires_grad_(any(n.startswith(q) for q in PREF) and "origin_attn" not in n and (not UP3 or n.startswith("up_blocks.3")))
TRAIN = [(n, p) for n, p in pipe.unet.named_parameters() if p.requires_grad]; print("trainable", len(TRAIN), flush=True)
configure_unet_memory_features(pipeline=pipe, enable_gradient_checkpointing=True, checkpoint_use_reentrant=False, attn_mode="auto", ff_chunk_size=(1 if FF else None), ff_chunk_dim=1)
print("FF", FF, "UP3", UP3, flush=True)
pipe.unet.train()
sched = EulerDiscreteScheduler.from_config(pipe.scheduler.config); sched.set_timesteps(8, device=dev)
def aligned_batch(clip, s, e):
    vr = VideoReader(f"video_data/train/{clip}_train.mp4", ctx=cpu(0))
    fr = torch.from_numpy(vr.get_batch(list(range(s, e))).asnumpy()).permute(0, 3, 1, 2).float() / 255.0
    H, W = fr.shape[2] // 2, fr.shape[3] // 2
    TR, BL, BR = fr[:, :, :H, W:], fr[:, :, H:, :W], fr[:, :, H:, W:]
    h, w = H // 128 * 128, W // 128 * 128
    TR, BL, BR = TR[:, :, :h, :w], BL[:, :, :h, :w], BR[:, :, :h, :w]
    top, left = (h - 576) // 2, (w - 1024) // 2
    sl = (slice(None), slice(None), slice(top, top + 576), slice(left, left + 1024))
    return BR[sl], BL[sl].mean(dim=1, keepdim=True), TR[sl]
@torch.no_grad()
def encode(cond, mask, target):
    cond, mask, target = cond.to(dev, dt), mask.to(dev, dt), target.to(dev, dt); H, W = cond.shape[2], cond.shape[3]
    emb = pipe._encode_image(cond[0:1].float(), device=dev, num_videos_per_prompt=1, do_classifier_free_guidance=False)
    fc = pipe.image_processor.preprocess(cond, height=H, width=W)
    lat = torch.cat([pipe.vae.encode(fc[i:i + 1]).latent_dist.mode() for i in range(fc.shape[0])], 0).unsqueeze(0)
    fm = pipe.mask_processor.preprocess(mask, height=H, width=W); ml = F.interpolate(fm, scale_factor=1 / pipe.vae_scale_factor).unsqueeze(0).to(dt)
    ft = pipe.image_processor.preprocess(target, height=H, width=W)
    x0 = torch.cat([pipe.vae.encode(ft[i:i + 1]).latent_dist.mode() for i in range(ft.shape[0])], 0).unsqueeze(0) * pipe.vae.config.scaling_factor
    return emb, lat.to(dt), ml, x0.to(dt), torch.tensor([[6.0, 127.0, 0.0]], dtype=dt, device=dev)
OUT = {}
for nf in [int(x) for x in sys.argv[1:]]:
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    emb, lat, ml, x0, add = encode(*aligned_batch("0301", 40, 40 + nf))
    enc_peak = torch.cuda.max_memory_allocated() / 2**30
    times = []
    for it in range(3):
        idx = 3  # sigma 31
        t = sched.timesteps[idx].to(torch.float32).reshape(1); sigma = sched.sigmas[idx].to(dt); den = (sigma ** 2 + 1).sqrt()
        eps = torch.randn_like(x0); x_t = ((x0 + eps * sigma) / den).to(dt); target = ((eps - sigma * x0) / den)
        for _, p in TRAIN: p.grad = None
        torch.cuda.synchronize(); t0 = time.time()
        pred = pipe.unet(torch.cat([x_t, lat, ml], dim=2), t, encoder_hidden_states=emb, added_time_ids=add, return_dict=False)[0]
        loss = (pred.float() - target.float()).pow(2).mean(); loss.backward()
        torch.cuda.synchronize(); times.append(time.time() - t0)
    OUT[nf] = {"loss": round(loss.item(), 4), "enc_peak_GiB": round(enc_peak, 2), "peak_alloc_GiB": round(torch.cuda.max_memory_allocated() / 2**30, 2),
               "peak_reserved_GiB": round(torch.cuda.max_memory_reserved() / 2**30, 2), "fwd_bwd_s": [round(x, 1) for x in times]}
    print(nf, json.dumps(OUT[nf]), flush=True)
    del emb, lat, ml, x0, pred, loss
json.dump(OUT, open(f"scripts/distill/runs/diag_trainer/xcheck_mem_window_ff{FF}_up3{int(UP3)}.json", "w"), indent=1)
