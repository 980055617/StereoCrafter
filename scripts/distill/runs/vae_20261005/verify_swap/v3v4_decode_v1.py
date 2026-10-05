#!/usr/bin/env python
"""vae_20261005 / verify_swap -- V3 decoder identity + V4 consistency-decoder determinism (PREREG.txt).  GPU, fresh process.

Written from the diffusers API (dswap_lib is NOT imported).  For <clip>_deliv_cap window <k> (a non-first window):
  * load the captured bf16 latent window (md5 vs latents_meta.json), z = (1/0.18215) * L on the GPU in bf16 (the deployed
    decode_latents expression), flatten frames;
  * kept local frames of window k by the keep rule of inpainting_inference.main (frames_chunk 14, overlap 3), re-derived here;
  * ftmse: AutoencoderKL (sd-vae-ft-mse) fp32, kept frames in pairs; stock: AutoencoderKLTemporalDecoder (SVD, fp16 variant -> bf16),
    whole window in chunks of 2 with num_frames=2, kept frames selected afterwards; cd: ConsistencyDecoderVAE fp16, 2 steps,
    Generator('cuda').manual_seed(seed) before every frame, cudnn deterministic -- decoded TWICE with seed 20261005, once with seed 1;
  * uint8 path: (x/2+0.5).clamp(0,1) -> round(x*255) (PIL values) -> /255 -> *255 -> truncating cast (the deployed path).
    Each is compared with the render's right half at the same global frames: bit-exact?  Also the round-only path.
  * DS-SEED-style region stats in render geometry from the model input BR (8-px-dilated hole excluded; EDGE = top decile of BR
    gradient, FLAT = bottom half, HOLE = dilated hole): mean |a-b| (RGB, [0,1]) for cd vs cd@1, cd vs stock, ftmse vs stock.
  * optional (--headroom): the same window of the real-right-eye latents (decoder_swap gt_lat_dev/<clip>) with ftmse vs
    headroom_dev/<clip>__ftmse.mkv.
usage: CUDA_VISIBLE_DEVICES=<g> flock /tmp/claude-gpu<g>.lock python v3v4_decode_v1.py <out_json> <clip> <k> [--headroom]
"""
import hashlib
import json
import os
import sys
import time

import cv2
import numpy as np
import torch
from decord import VideoReader, cpu
from diffusers import AutoencoderKL, ConsistencyDecoderVAE
from diffusers.models.autoencoders.autoencoder_kl_temporal_decoder import AutoencoderKLTemporalDecoder

REPO = "/home/kawa/master_project/StereoCrafter"
os.chdir(REPO)
OUT, CLIP, K = sys.argv[1], sys.argv[2], int(sys.argv[3])
DO_HR = "--headroom" in sys.argv[4:]
assert not os.path.exists(OUT), f"refusing to overwrite {OUT}"
SVD = "weights/stable-video-diffusion-img2vid-xt-1-1/"
FTMSE = "/home/kawa/master_project/third_party/DiffuEraser/weights/sd-vae-ft-mse"
CDP = ("/mnt/ssd_data/vae_20261005/decoder_swap/hf_home/hub/models--openai--consistency-decoder/snapshots/"
       "63b7a48896d92b6f56772f4111d0860b1bee3dd3")
LAT = f"/mnt/ssd_data/deep_20261004/decoder_ft/latents/{CLIP}_deliv_cap"
RED = "/mnt/ssd_data/vae_20261005/decoder_swap/redec_dev"
GTL = f"/mnt/ssd_data/vae_20261005/decoder_swap/gt_lat_dev/{CLIP}"
HRD = "/mnt/ssd_data/vae_20261005/decoder_swap/headroom_dev"
SF = 0.18215
T0 = time.time()


def log(*a):
    print(f"[v3v4 {CLIP} w{K} {time.time() - T0:6.1f}s]", *a, flush=True)


# ------------------------------------------------------------------ window / keep rule (re-derived)
n = len(VideoReader(f"video_data/splatting/{CLIP}_splatting_results.mp4", ctx=cpu(0)))
wins, started = [], False
for i in range(0, n, 14 - 3):
    if i + 3 >= n:
        break
    if started and i + 14 > n:
        ci = max(n + 3 - 14, 0)
        cov = i - ci + 3
    else:
        ci, cov = i, 3
    wl = min(14, n - ci)
    wins.append((i, ci, cov, wl))
    started = True
i, ci, cov, wl = wins[K]
assert K > 0, "PREREG: a non-first window"
local = list(range(cov, wl))
glob = [ci + p for p in local]
log(f"n {n}, {len(wins)} windows; window {K}: start {ci}, overlap {cov}, len {wl}; kept local {local[0]}..{local[-1]} -> global {glob[0]}..{glob[-1]}")


def load_lat(d, k):
    meta = json.load(open(f"{d}/latents_meta.json"))
    lat = torch.load(f"{d}/w{k:03d}.pt", map_location="cpu")
    md5 = hashlib.md5(lat.contiguous().view(torch.int16).numpy().tobytes()).hexdigest()
    assert md5 == meta["windows"][k]["md5"], (d, k, md5)
    assert lat.dtype == torch.bfloat16 and lat.shape[1] == wl, (lat.dtype, lat.shape)
    return lat.to("cuda"), md5


def to_u8(x):
    """x: float32 [F,3,H,W] in [-1,1] on GPU -> (deployed uint8 [F,H,W,3], round-only uint8)."""
    y = (x / 2 + 0.5).clamp(0, 1).permute(0, 2, 3, 1).float().cpu().numpy()
    u = (y * 255).round().astype(np.uint8)                       # PIL values (VaeImageProcessor.numpy_to_pil)
    dep = ((u.astype(np.float32) / np.float32(255.0)) * np.float32(255.0)).astype(np.uint8)   # /255 then *255, truncating
    return dep, u


def render_right(path, frames):
    vr = VideoReader(path, ctx=cpu(0))
    a = vr.get_batch(frames).asnumpy()
    return np.ascontiguousarray(a[:, :, a.shape[2] // 2:]) if a.shape[2] == 2048 else np.ascontiguousarray(a)


def cmp(a, b):
    d = np.abs(a.astype(np.int16) - b.astype(np.int16))
    return dict(equal=bool(np.array_equal(a, b)), n_diff=int((d > 0).sum()), max_abs=int(d.max()), frac_diff=float((d > 0).mean()))


res = dict(clip=CLIP, window=K, start=ci, overlap=cov, win_len=wl, kept_local=local, kept_global=glob,
           gpu=os.environ.get("CUDA_VISIBLE_DEVICES"), torch=torch.__version__, cudnn=torch.backends.cudnn.version())
lat, res["latent_md5"] = load_lat(LAT, K)
z = (1 / SF) * lat                                                  # bf16, the deployed expression
z = z.flatten(0, 1)
out = {}
with torch.no_grad():
    # ---------------- ftmse
    vae = AutoencoderKL.from_pretrained(FTMSE, torch_dtype=torch.float32).to("cuda").eval()
    zz = z[local].to(torch.float32)
    x = torch.cat([vae.decode(zz[j:j + 2]).sample.float() for j in range(0, zz.shape[0], 2)], 0)
    out["ftmse"] = to_u8(x)
    if DO_HR:
        glat, res["gt_latent_md5"] = load_lat(GTL, K)
        gz = ((1 / SF) * glat).flatten(0, 1)[local].to(torch.float32)
        gx = torch.cat([vae.decode(gz[j:j + 2]).sample.float() for j in range(0, gz.shape[0], 2)], 0)
        hdep, hround = to_u8(gx)
        ref = render_right(f"{HRD}/{CLIP}__ftmse.mkv", glob)
        res["headroom_ftmse_vs_mkv"] = dict(deployed_path=cmp(hdep, ref), round_only=cmp(hround, ref))
        log(f"headroom ftmse (real-right-eye latents) vs headroom_dev mkv: {res['headroom_ftmse_vs_mkv']}")
    del vae
    torch.cuda.empty_cache()
    # ---------------- stock (SVD temporal decoder, bf16, chunks of 2 over the whole window)
    svae = AutoencoderKLTemporalDecoder.from_pretrained(SVD, subfolder="vae", variant="fp16", torch_dtype=torch.bfloat16)
    svae = svae.to(dtype=torch.bfloat16).to("cuda").eval()
    fr = [svae.decode(z[j:j + 2], num_frames=z[j:j + 2].shape[0]).sample for j in range(0, z.shape[0], 2)]
    xs = torch.cat(fr, 0).float()[local]
    out["stock"] = to_u8(xs)
    del svae
    torch.cuda.empty_cache()
    # ---------------- cd: twice with seed 20261005, once with seed 1
    cvae = ConsistencyDecoderVAE.from_pretrained(CDP, torch_dtype=torch.float16).to("cuda").eval()
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    zc = z[local].to(torch.float16)

    def cd(seed):
        o = []
        for j in range(zc.shape[0]):
            g = torch.Generator(device="cuda").manual_seed(seed)
            y = cvae.decode(zc[j:j + 1], generator=g, num_inference_steps=2).sample.float()
            assert torch.isfinite(y).all()
            o.append(y)
        return torch.cat(o, 0)

    c1 = cd(20261005)
    c2 = cd(20261005)
    c3 = cd(1)
    res["cd_float_run1_eq_run2"] = bool(torch.equal(c1, c2))
    out["cd"], out["cd_run2"], out["cd_seed1"] = to_u8(c1), to_u8(c2), to_u8(c3)
    del cvae

res["identity"] = {}
for name, row in (("ftmse", "ftmse"), ("stock", "stock"), ("cd", "cd")):
    ref = render_right(f"{RED}/{CLIP}_deliv_cap__{row}/{CLIP}_inpainting_results_sbs.mkv", glob)
    rj = json.load(open(f"{RED}/{CLIP}_deliv_cap__{row}/redecode.json"))
    res["identity"][name] = dict(render_gpu=rj["gpu"], deployed_path=cmp(out[name][0], ref), round_only=cmp(out[name][1], ref))
    log(f"{name:6s} (render made on GPU {rj['gpu']}, this run GPU {res['gpu']}): {res['identity'][name]}")
res["cd_run1_vs_run2_uint8"] = cmp(out["cd"][0], out["cd_run2"][0])
log(f"cd run1 vs run2: float equal {res['cd_float_run1_eq_run2']}, uint8 {res['cd_run1_vs_run2_uint8']}")

# ------------------------------------------------------------------ DS-SEED-style region stats (model-input regions)
vs = VideoReader(f"video_data/splatting/{CLIP}_splatting_results.mp4", ctx=cpu(0))
S = vs.get_batch(glob).asnumpy()
Hs, Ws = S.shape[1] // 2, S.shape[2] // 2
st0, sl0 = (Hs // 128 * 128 - 576) // 2, (Ws // 128 * 128 - 1024) // 2
BR = S[:, Hs + st0:Hs + st0 + 576, Ws + sl0:Ws + sl0 + 1024]
HOLE = S[:, Hs + st0:Hs + st0 + 576, sl0:sl0 + 1024].astype(np.float32).mean(-1) > 127.5
ker = np.ones((17, 17), np.uint8)
HD = np.stack([cv2.dilate(h.astype(np.uint8), ker) > 0 for h in HOLE])
g = BR.astype(np.float32).mean(-1) / 255.
gm = np.zeros_like(g)
gm[:, :, :-1] += np.abs(np.diff(g, axis=2))
gm[:, :-1, :] += np.abs(np.diff(g, axis=1))
valid = ~HD
REG = dict(edge=valid & (gm >= np.quantile(gm[valid], 0.90)), flat=valid & (gm <= np.quantile(gm[valid], 0.50)), hole=HD)


def mad(a, b):
    d = np.abs(a.astype(np.float32) - b.astype(np.float32)).mean(-1) / 255.
    return {r: (float(d[m].mean()) if m.any() else None) for r, m in REG.items()} | {"all": float(d.mean())}


res["region_frac"] = {r: float(m.mean()) for r, m in REG.items()}
res["seed_stats"] = {"cd vs cd@1": mad(out["cd"][0], out["cd_seed1"][0]), "cd vs stock": mad(out["cd"][0], out["stock"][0]),
                     "ftmse vs stock": mad(out["ftmse"][0], out["stock"][0])}
e1, e2 = res["seed_stats"]["cd vs cd@1"]["edge"], res["seed_stats"]["cd vs stock"]["edge"]
res["seed_claim_holds"] = bool(e1 >= 0.8 * e2)
for k_, v_ in res["seed_stats"].items():
    log(f"{k_:16s} " + "  ".join(f"{r} {v_[r]:.5f}" if v_[r] is not None else f"{r} n/a" for r in ("edge", "flat", "hole", "all")))
log(f"claim 'cd edge change vs stock is as large as the seed change' (seed >= 0.8 x stock-diff at edges): {res['seed_claim_holds']}"
    f" ({e1:.5f} vs {e2:.5f})")
np.savez_compressed(f"/mnt/ssd_data/vae_20261005/verify_swap/v3v4_{CLIP}_w{K}.npz", frames=np.array(glob),
                    **{k_: v_[0] for k_, v_ in out.items()})
res["seconds"] = time.time() - T0
json.dump(res, open(OUT, "w"), indent=1)
print("V3V4_DONE", CLIP, K, flush=True)
