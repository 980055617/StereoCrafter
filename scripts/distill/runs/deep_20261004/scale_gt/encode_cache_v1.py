#!/usr/bin/env python
"""scale_gt lane, PHASE B (GPU, run under flock /tmp/claude-gpu1.lock): latent cache for the GT-supervised scale run.

For every window dir written by prep_crops_v2.py (<crops_root>/<clip>/w<start>/{cond,bl,tgt,valid}.npy) compute EXACTLY the
tensors scripts/distill/runs/diag_trainer/minift/xcheck_mini_ft.py builds in its WINS list (its encode() is copied verbatim):
  emb  = CLIP image embedding of cond frame 0           lat = RAW VAE latents of the cond (BR) frames (x1.0, as the pipeline)
  ml   = mask latent (BL 3-channel mean, mask_processor, 1/8 interpolate)
  x0   = VAE latents of the REGISTERED real right eye * scaling_factor        add = [[6, 127, 0]]
plus valid [1,14,1,72,128] bool (the validity mask on the latent grid).  One .pt per window:
  <out_dir>/<clip>_w<start:03d>.pt ; an existing file is never overwritten (skipped).
The md5s of the source arrays are re-checked against clip.json before encoding.
usage: CUDA_VISIBLE_DEVICES=1 flock /tmp/claude-gpu1.lock python encode_cache_v1.py <crops_root> <out_dir> <clips_json_or_comma_list>
"""
import hashlib, json, os, sys, time
REPO = "/home/kawa/master_project/StereoCrafter"; os.chdir(REPO); sys.path.insert(0, REPO)
os.environ.setdefault("MAMBA_SELF_ATTN_INCLUDE", "__nomatch__")
import numpy as np
import torch, torch.nn.functional as F
from transformers import CLIPVisionModelWithProjection
from diffusers.models.unets.unet_spatio_temporal_condition import UNetSpatioTemporalConditionModel
from diffusers.models.autoencoders.autoencoder_kl_temporal_decoder import AutoencoderKLTemporalDecoder
from pipelines.stereo_video_inpainting import StableVideoDiffusionInpaintingPipeline as _Pipe

CROPS, OUT, CL = sys.argv[1], sys.argv[2], sys.argv[3]
clips = json.load(open(CL))["clips"] if CL.endswith(".json") else CL.split(",")
os.makedirs(OUT, exist_ok=True)
dev = torch.device("cuda:0"); dt = torch.bfloat16
pre = "weights/stable-video-diffusion-img2vid-xt-1-1/"; unet_path = "weights/StereoCrafter/"
image_encoder = CLIPVisionModelWithProjection.from_pretrained(pre, subfolder="image_encoder", variant="fp16", torch_dtype=dt)
vae = AutoencoderKLTemporalDecoder.from_pretrained(pre, subfolder="vae", variant="fp16", torch_dtype=dt)
unet = UNetSpatioTemporalConditionModel.from_pretrained(unet_path, subfolder="unet_diffusers", low_cpu_mem_usage=True, torch_dtype=dt)
pipe = _Pipe.from_pretrained(pre, image_encoder=image_encoder, vae=vae, unet=unet, torch_dtype=dt).to(dev)
pipe.vae.eval(); pipe.image_encoder.eval(); pipe.unet.eval()


@torch.no_grad()
def encode(cond, mask, target):   # == xcheck_mini_ft.py encode(), verbatim
    cond, mask, target = cond.to(dev, dt), mask.to(dev, dt), target.to(dev, dt); H, W = cond.shape[2], cond.shape[3]
    emb = pipe._encode_image(cond[0:1].float(), device=dev, num_videos_per_prompt=1, do_classifier_free_guidance=False)
    fc = pipe.image_processor.preprocess(cond, height=H, width=W)
    lat = torch.cat([pipe.vae.encode(fc[i:i + 1]).latent_dist.mode() for i in range(fc.shape[0])], 0).unsqueeze(0)   # RAW cond latents (x1.0), as the pipeline
    fm = pipe.mask_processor.preprocess(mask, height=H, width=W); ml = F.interpolate(fm, scale_factor=1 / pipe.vae_scale_factor).unsqueeze(0).to(dt)
    ft = pipe.image_processor.preprocess(target, height=H, width=W)
    x0 = torch.cat([pipe.vae.encode(ft[i:i + 1]).latent_dist.mode() for i in range(ft.shape[0])], 0).unsqueeze(0) * pipe.vae.config.scaling_factor
    return emb, lat.to(dt), ml, x0.to(dt), torch.tensor([[6.0, 127.0, 0.0]], dtype=dt, device=dev)


def md5(a):
    return hashlib.md5(np.ascontiguousarray(a).tobytes()).hexdigest()


ONLY = None
if os.environ.get("ENC_SPEC"):   # optional: encode only the windows listed in these training spec(s) (comma list of spec jsons)
    ONLY = {(c, int(s)) for sp in os.environ["ENC_SPEC"].split(",") for c, s in json.load(open(sp))["windows"]}
    print(f"[enc] restricted to {len(ONLY)} windows from ENC_SPEC", flush=True)
t_all = time.time(); n_done = n_skip = n_notlisted = 0
for clip in clips:
    cj = json.load(open(os.path.join(CROPS, clip, "clip.json")))
    for ws in cj["windows_stats"]:
        s = ws["start"]; outp = os.path.join(OUT, f"{clip}_w{s:03d}.pt")
        if ONLY is not None and (clip, s) not in ONLY:
            n_notlisted += 1; continue
        if os.path.exists(outp):
            n_skip += 1; continue
        wd = os.path.join(CROPS, clip, f"w{s:03d}")
        arr = {k: np.load(os.path.join(wd, f"{k}.npy")) for k in ("cond", "bl", "tgt", "valid")}
        for k in arr:
            assert md5(arr[k]) == ws["md5"][k], f"md5 mismatch {clip} w{s} {k}"
        t0 = time.time()
        BR = torch.from_numpy(arr["cond"]).permute(0, 3, 1, 2).float() / 255.0
        M = (torch.from_numpy(arr["bl"]).permute(0, 3, 1, 2).float() / 255.0).mean(dim=1, keepdim=True)
        TR = torch.from_numpy(arr["tgt"]).permute(0, 3, 1, 2).float() / 255.0
        emb, lat, ml, x0, add = encode(BR, M, TR)
        valid = torch.from_numpy(arr["valid"]).bool()[None, :, None]          # [1,14,1,72,128]
        assert valid.shape[-2:] == x0.shape[-2:] and valid.shape[1] == x0.shape[1], (valid.shape, x0.shape)
        torch.save(dict(emb=emb.cpu(), lat=lat.cpu(), ml=ml.cpu(), x0=x0.cpu(), add=add.cpu(), valid=valid,
                        meta=dict(clip=clip, start=s, kept=ws["kept"], hole_frac=ws["hole_frac"], mask_frac=float(M.mean()),
                                  reg=[cj["ddy"], cj["ddx"]], src_md5=ws["md5"])), outp)
        n_done += 1
        if n_done % 20 == 1:
            print(f"[enc] {clip} w{s:03d} lat {tuple(lat.shape)} x0 {tuple(x0.shape)} ml mean {ml.float().mean():.4f} kept {ws['kept']:.3f} "
                  f"{time.time()-t0:.2f}s (done {n_done}, skipped {n_skip}, {time.time()-t_all:.0f}s) peak {torch.cuda.max_memory_allocated()/2**30:.1f} GiB", flush=True)
print(f"ENCODE_DONE done={n_done} skipped={n_skip} not_listed={n_notlisted} {time.time()-t_all:.0f}s", flush=True)
