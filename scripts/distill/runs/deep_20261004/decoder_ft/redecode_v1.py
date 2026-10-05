#!/usr/bin/env python
"""deep_20261004 / decoder_ft lane -- RE-DECODE captured pre-decode latents with a given VAE decoder, deployment-faithfully.

For each (clip, latent label) the captured windows ($LATR/<clip>_<latlabel>/w<k>.pt, written by capture_hook_v1.py) are
decoded with the pipeline's OWN functions, called on a namespace holding the same objects the pipeline holds:
  MambaStableVideoDiffusionInpaintingPipeline.decode_latents(shim, lat, num_frames=lat.shape[1], decode_chunk_size=2)
  -> tensor2vid(.., output_type="pil")[0] -> torch.tensor(np.array(img)).permute(2,0,1).float()/255
  -> keep rule of inpainting_inference.main (14-frame windows, overlap 3, generated[cur_overlap:] for i != 0)
  -> SBS: left half = the captured render's left half (lossless FFV1 read, identical by construction),
          right half = ((frames_output * 255).to(uint8)) exactly as main() does
  -> FFV1 SBS mkv via beyond4/infer_lossless.py _ffv1_write (+ .md5 of the pre-encode array, + writer_md5.txt)
VAE loading verbatim from inpainting_inference.main (variant fp16 -> bf16, requires_grad False, cuda).  A decoder
checkpoint (a state dict of vae.decoder.* in fp32 masters, written by train_decoder_v1.py) is copied INTO that VAE's
decoder with .to(bf16), so the stock row and every fine-tuned row share all other code.
Overlap_prev_weight is 0.0 in the deployed config, so decoded pixels never feed later windows: this IS the deployed output
for that decoder (no UNet call is repeated).

usage: CUDA_VISIBLE_DEVICES=1 flock /tmp/claude-gpu1.lock python redecode_v1.py <out_root> <dec_spec> <clip:latlabel>[,...]
  dec_spec = "stock" or "<name>=<path to decoder state dict .pt>" (several, comma separated: name1=p1,name2=p2)
  writes <out_root>/<clip>_<latlabel>__<name>/<clip>_inpainting_results_sbs.mkv (+ .md5, writer_md5.txt, redecode.json)
  an existing output dir is never overwritten (skipped with a message).
env: DF_LATR (default /mnt/ssd_data/deep_20261004/decoder_ft/latents), DF_CAPR (default .../decoder_ft/capture/clips)
"""
import hashlib
import importlib.util
import json
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

REPO = "/home/kawa/master_project/StereoCrafter"
sys.path.insert(0, REPO)
os.chdir(REPO)
_spec = importlib.util.spec_from_file_location(
    "infer_lossless", f"{REPO}/scripts/distill/runs/fulldata_v2/beyond4/infer_lossless.py")
_IL = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_IL)
import inpainting_inference as II  # noqa: E402
from diffusers.image_processor import VaeImageProcessor  # noqa: E402
from diffusers.models.autoencoders.autoencoder_kl_temporal_decoder import AutoencoderKLTemporalDecoder  # noqa: E402
from decord import VideoReader, cpu  # noqa: E402

PRE = "weights/stable-video-diffusion-img2vid-xt-1-1/"
LATR = os.environ.get("DF_LATR", "/mnt/ssd_data/deep_20261004/decoder_ft/latents")
CAPR = os.environ.get("DF_CAPR", "/mnt/ssd_data/deep_20261004/decoder_ft/capture/clips")
FC, OV, DCS = 14, 3, 2
dt = torch.bfloat16
T0 = time.time()


def log(*a):
    print(f"[redecode {time.time() - T0:7.1f}s]", *a, flush=True)


def load_vae():
    vae = AutoencoderKLTemporalDecoder.from_pretrained(PRE, subfolder="vae", variant="fp16", torch_dtype=dt)
    vae.requires_grad_(False)
    vae.to(dtype=dt)
    vae = vae.to("cuda")
    vae.eval()
    sf = 2 ** (len(vae.config.block_out_channels) - 1)
    shim = SimpleNamespace(vae=vae, vae_scale_factor=sf, image_processor=VaeImageProcessor(vae_scale_factor=sf))
    return shim


def stock_decoder_state(shim):
    return {k: v.detach().clone() for k, v in shim.vae.decoder.state_dict().items()}


def install_decoder(shim, sd):
    own = shim.vae.decoder.state_dict()
    assert set(sd) == set(own), (sorted(set(sd) ^ set(own)))[:5]
    with torch.no_grad():
        for k, p in own.items():
            p.copy_(sd[k].to(device=p.device, dtype=p.dtype))


def windows(n):
    """(i, cur_i, cur_overlap) for every window of inpainting_inference.main with frames_chunk 14, overlap 3."""
    out, generated = [], False
    for i in range(0, n, FC - OV):
        if i + OV >= n:
            break
        if generated and i + FC > n:
            cur_i = max(n + OV - FC, 0)
            cur_ov = i - cur_i + OV
        else:
            cur_i, cur_ov = i, OV
        out.append((i, cur_i, cur_ov))
        generated = True
    return out


@torch.no_grad()
def decode_clip(shim, clip, latlabel):
    ld = f"{LATR}/{clip}_{latlabel}"
    meta = json.load(open(f"{ld}/latents_meta.json"))
    vr = VideoReader(f"video_data/splatting/{clip}_splatting_results.mp4", ctx=cpu(0))
    n, fps = len(vr), float(vr.get_avg_fps())
    del vr
    W = windows(n)
    assert len(W) == meta["n_windows"], (clip, latlabel, len(W), meta["n_windows"])
    results = []
    for k, (i, cur_i, cur_ov) in enumerate(W):
        lat = torch.load(f"{ld}/w{k:03d}.pt", map_location="cpu")
        raw = lat.contiguous().view(torch.int16) if lat.element_size() == 2 else lat
        assert hashlib.md5(raw.numpy().tobytes()).hexdigest() == meta["windows"][k]["md5"], (clip, latlabel, k)
        lat = lat.to("cuda")
        assert lat.dtype == shim.vae.dtype, (lat.dtype, shim.vae.dtype)
        vf = II._Pipe.decode_latents(shim, lat, num_frames=lat.shape[1], decode_chunk_size=DCS)
        vf = II.tensor2vid(vf, shim.image_processor, output_type="pil")[0]
        g = torch.stack([torch.tensor(np.array(im)).permute(2, 0, 1).to(dtype=torch.float32) / 255.0 for im in vf])
        if i != 0:
            g = g[cur_ov:]
        results.append(g)
    out = torch.cat(results, dim=0)
    assert out.shape[0] == n, (out.shape, n)
    right = (out * 255).permute(0, 2, 3, 1).to(dtype=torch.uint8).cpu().numpy()
    return right, fps, n


def left_half(clip, latlabel, n):
    p = f"{CAPR}/{clip}_{latlabel}/{clip}_inpainting_results_sbs.mkv"
    vr = VideoReader(p, ctx=cpu(0))
    assert len(vr) == n, (p, len(vr), n)
    a = vr.get_batch(list(range(n))).asnumpy()
    return np.ascontiguousarray(a[:, :, : a.shape[2] // 2]), p


def main():
    out_root, dec_spec, pairs = sys.argv[1], sys.argv[2], [x.split(":") for x in sys.argv[3].split(",")]
    os.makedirs(out_root, exist_ok=True)
    decs = [("stock", None)] if dec_spec == "stock" else [tuple(x.split("=", 1)) for x in dec_spec.split(",")]
    shim = load_vae()
    stock = stock_decoder_state(shim)
    log(f"VAE loaded dtype={shim.vae.dtype}; decoders {[d[0] for d in decs]}; pairs {pairs}")
    lefts = {}
    for name, path in decs:
        if path is None:
            install_decoder(shim, stock)
            dmd5 = "stock"
        else:
            sd = torch.load(path, map_location="cpu", weights_only=False)   # this lane's own checkpoint (config holds a TorchVersion)
            sd = sd.get("decoder", sd)
            install_decoder(shim, sd)
            dmd5 = hashlib.md5(open(path, "rb").read()).hexdigest()
        ndiff = sum(int(not torch.equal(shim.vae.decoder.state_dict()[k], stock[k])) for k in stock)
        log(f"decoder {name} ({path}) md5 {dmd5}: {ndiff}/{len(stock)} bf16 tensors differ from stock")
        for clip, latlabel in pairs:
            od = f"{out_root}/{clip}_{latlabel}__{name}"
            if os.path.exists(od):
                log(f"SKIP {od} exists")
                continue
            t0 = time.time()
            right, fps, n = decode_clip(shim, clip, latlabel)
            if (clip, latlabel) not in lefts:
                lefts[(clip, latlabel)] = left_half(clip, latlabel, n)
            left, lp = lefts[(clip, latlabel)]
            sbs = np.ascontiguousarray(np.concatenate([left, right], axis=2))
            os.makedirs(od)
            p = f"{od}/{clip}_inpainting_results_sbs.mkv"
            dig = hashlib.md5(sbs.tobytes()).hexdigest()
            _IL._ffv1_write(sbs, fps, p)
            with open(p + ".md5", "w") as fh:
                fh.write(f"{dig}  {tuple(sbs.shape)}  fps={float(fps):.6f}\n")
            with open(f"{od}/writer_md5.txt", "w") as fh:
                fh.write(f"{dig} {tuple(sbs.shape)} uint8 {clip}_inpainting_results_sbs.mp4\n")
            json.dump(dict(clip=clip, latlabel=latlabel, decoder=name, decoder_path=path, decoder_md5=dmd5,
                           n_tensors_differ_from_stock=ndiff, left_from=lp, n=n, fps=fps, md5=dig,
                           seconds=time.time() - t0), open(f"{od}/redecode.json", "w"), indent=1)
            log(f"wrote {p} md5={dig} n={n} ({time.time() - t0:.1f}s)")
    log("REDECODE_DONE")


if __name__ == "__main__":
    main()
