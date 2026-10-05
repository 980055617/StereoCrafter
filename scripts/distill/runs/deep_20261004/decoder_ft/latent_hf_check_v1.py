#!/usr/bin/env python
"""deep_20261004 / decoder_ft lane -- LATENT high-frequency check (post-hoc mechanism diagnostic, dev clips only; it is
not a selection input and changes no verdict).  Question: do the models' sampled latents carry the detail that the latent of
their own input carries?  If the output latents lack it, the deployed blur is upstream of the decoder; if they carry it
(and/or carry extra flat-region energy), the decoder sees a different latent distribution than the one it was tuned on.

Per dev clip, per captured window k (latents/<clip>_{origin,deliv}_cap/w<k>.pt, bf16 [1,n,4,72,128], x0 space = mode*sf):
  INPUT  = the deployed encode of the model's own conditioning frames (warped BR crop of the splatting video, deployed window,
           image_processor.preprocess -> _encode_vae_frames(n_frames_per_time=5) mode) * 0.18215  (same space as the outputs;
           geometry-aligned with the outputs by construction)
  per-cell HF = mean over the 4 channels of |4-neighbour Laplacian| of the latent (interior cells)
  cells: HOLE = any disocclusion pixel (BL mask > 127.5) in the 8x8 cell, dilated by 1 cell -> excluded everywhere;
         EDGE = top 20 % of non-hole cells by the INPUT frame's mean |gradient| over the cell; FLAT = bottom 50 %.
Reported: mean HF on EDGE and FLAT cells for INPUT / origin / deliverable, and the ratios output/INPUT, pooled over all
windows and frames; plus the same for the real right eye's own encode (deployed window, not aligned: whole-frame reference).
usage: CUDA_VISIBLE_DEVICES=1 flock /tmp/claude-gpu1.lock python latent_hf_check_v1.py <out_json>
"""
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

REPO = "/home/kawa/master_project/StereoCrafter"
sys.path.insert(0, REPO)
os.chdir(REPO)
import inpainting_inference as II  # noqa: E402
from diffusers.image_processor import VaeImageProcessor  # noqa: E402
from diffusers.models.autoencoders.autoencoder_kl_temporal_decoder import AutoencoderKLTemporalDecoder  # noqa: E402
from decord import VideoReader, cpu  # noqa: E402

OUTJ = sys.argv[1]
assert not os.path.exists(OUTJ), OUTJ
LATR = "/mnt/ssd_data/deep_20261004/decoder_ft/latents"
DEV = "0040 0082 0091 0184 0245 0268".split()
TH, TW, FC, OV, SF = 576, 1024, 14, 3, 0.18215
dt = torch.bfloat16
vae = AutoencoderKLTemporalDecoder.from_pretrained("weights/stable-video-diffusion-img2vid-xt-1-1/", subfolder="vae",
                                                   variant="fp16", torch_dtype=dt)
vae.requires_grad_(False)
vae = vae.to("cuda").eval()
shim = SimpleNamespace(vae=vae, vae_scale_factor=8, image_processor=VaeImageProcessor(vae_scale_factor=8))
LAP = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32, device="cuda").view(1, 1, 3, 3)


def windows(n):
    out, gen = [], False
    for i in range(0, n, FC - OV):
        if i + OV >= n:
            break
        if gen and i + FC > n:
            cur_i = max(n + OV - FC, 0)
            cur_ov = i - cur_i + OV
        else:
            cur_i, cur_ov = i, OV
        out.append((i, cur_i, cur_ov))
        gen = True
    return out


def hf(z):
    """z float [n,4,72,128] -> per-cell HF [n,70,126] (interior)"""
    n = z.shape[0]
    y = F.conv2d(z.reshape(n * 4, 1, 72, 128).float(), LAP)
    return y.abs().reshape(n, 4, 70, 126).mean(1)


@torch.no_grad()
def encode(frames_u8):
    x = torch.from_numpy(frames_u8).permute(0, 3, 1, 2).float() / 255.0
    xp = shim.image_processor.preprocess(x, height=TH, width=TW)
    z = II._Pipe._encode_vae_frames(shim, xp, torch.device("cuda"), 1, False, n_frames_per_time=5)[0]
    return (z * SF).float()


res = dict(clips={}, definition=__doc__)
pool = {k: {"EDGE": [], "FLAT": []} for k in ("INPUT", "origin", "deliv")}
gt_all = []
with torch.no_grad():
    for clip in DEV:
        vs = VideoReader(f"video_data/splatting/{clip}_splatting_results.mp4", ctx=cpu(0))
        n = len(vs)
        s0 = vs[0].asnumpy()
        Hs, Ws = s0.shape[0] // 2, s0.shape[1] // 2
        st0, sl0 = (Hs // 128 * 128 - TH) // 2, (Ws // 128 * 128 - TW) // 2
        W_ = windows(n)
        cl = {k: {"EDGE": [], "FLAT": []} for k in ("INPUT", "origin", "deliv")}
        for k, (i, cur_i, cur_ov) in enumerate(W_):
            idx = list(range(cur_i, min(cur_i + FC, n)))
            fr = vs.get_batch(idx).asnumpy()
            BR = np.ascontiguousarray(fr[:, Hs + st0:Hs + st0 + TH, Ws + sl0:Ws + sl0 + TW])
            BL = fr[:, Hs + st0:Hs + st0 + TH, sl0:sl0 + TW].astype(np.float32).mean(-1) > 127.5
            zin = encode(BR)
            zo = torch.load(f"{LATR}/{clip}_origin_cap/w{k:03d}.pt").to("cuda")[0].float()
            zd = torch.load(f"{LATR}/{clip}_deliv_cap/w{k:03d}.pt").to("cuda")[0].float()
            assert zo.shape == zin.shape == zd.shape, (zo.shape, zin.shape)
            hole = torch.from_numpy(BL).float().cuda().unsqueeze(1)
            hc = F.max_pool2d(hole, 8) > 0                                    # [n,1,72,128] any hole pixel in the cell
            hc = F.max_pool2d(hc.float(), 3, 1, 1) > 0                        # dilate by one cell
            g = torch.from_numpy(BR).float().cuda().mean(-1)
            gm = torch.zeros_like(g)
            gm[:, :, :-1] += (g[:, :, 1:] - g[:, :, :-1]).abs()
            gm[:, :-1, :] += (g[:, 1:, :] - g[:, :-1, :]).abs()
            gc = F.avg_pool2d(gm.unsqueeze(1), 8)                             # [n,1,72,128]
            valid = ~hc[:, 0, 1:-1, 1:-1]
            gci = gc[:, 0, 1:-1, 1:-1]
            v = gci[valid]
            if v.numel() < 100:
                continue
            hi, lo = torch.quantile(v, 0.80), torch.quantile(v, 0.50)
            edge = valid & (gci >= hi)
            flat = valid & (gci <= lo)
            for name, z in (("INPUT", zin), ("origin", zo), ("deliv", zd)):
                h = hf(z)
                cl[name]["EDGE"].append(float(h[edge].mean()))
                cl[name]["FLAT"].append(float(h[flat].mean()))
        # real right eye of the dev clip (deployed window, unregistered): whole-frame reference level
        TR = np.load(f"/mnt/ssd_data/deep_20261004/decoder_ft/gt_dev/{clip}_TR.npy", mmap_mode="r")
        hs = []
        for s in range(0, min(len(TR), 140), 14):
            hs.append(float(hf(encode(np.ascontiguousarray(TR[s:s + 14]))).mean()))
        out = {nm: {c: float(np.mean(cl[nm][c])) for c in ("EDGE", "FLAT")} for nm in cl}
        out["ratio"] = {f"{m}/INPUT": {c: out[m][c] / out["INPUT"][c] for c in ("EDGE", "FLAT")} for m in ("origin", "deliv")}
        out["GT_wholeframe_HF"] = float(np.mean(hs))
        out["INPUT_wholeframe_note"] = "GT reference is not geometry-aligned; compare levels only"
        res["clips"][clip] = out
        for nm in cl:
            for c in ("EDGE", "FLAT"):
                pool[nm][c].append(out[nm][c])
        gt_all.append(out["GT_wholeframe_HF"])
        print(f"{clip}: EDGE in {out['INPUT']['EDGE']:.4f} origin {out['origin']['EDGE']:.4f} deliv {out['deliv']['EDGE']:.4f} "
              f"(ratios {out['ratio']['origin/INPUT']['EDGE']:.3f} / {out['ratio']['deliv/INPUT']['EDGE']:.3f}) | FLAT in "
              f"{out['INPUT']['FLAT']:.4f} origin {out['origin']['FLAT']:.4f} deliv {out['deliv']['FLAT']:.4f} (ratios "
              f"{out['ratio']['origin/INPUT']['FLAT']:.3f} / {out['ratio']['deliv/INPUT']['FLAT']:.3f}) | GT whole-frame HF "
              f"{out['GT_wholeframe_HF']:.4f}", flush=True)
res["mean"] = {nm: {c: float(np.mean(pool[nm][c])) for c in ("EDGE", "FLAT")} for nm in pool}
res["mean_ratio"] = {m: {c: float(np.mean([res["clips"][cc]["ratio"][f"{m}/INPUT"][c] for cc in res["clips"]]))
                         for c in ("EDGE", "FLAT")} for m in ("origin", "deliv")}
json.dump(res, open(OUTJ, "w"), indent=1)
print("MEAN", json.dumps(res["mean"]), "RATIO", json.dumps(res["mean_ratio"]), flush=True)
