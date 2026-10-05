#!/usr/bin/env python
"""vae_20261005 / decoder_swap -- STEP 3 HEADROOM decodes (PREREG.txt section 2).  GPU, under the GPU lock.

(A) GT latents, once per clip: real right eye at the deployed window (decoder_ft gt_dev/<clip>_TR.npy, read only) ->
    the deployed encode per deployed 14/3 window (VAE fp16 variant -> bf16, image_processor.preprocess,
    _encode_vae_frames n_frames_per_time=5, mode) -> bf16(z * sf) [roundtrip_v1.py verbatim] ->
    /mnt/ssd_data/vae_20261005/decoder_swap/gt_lat_dev/<clip>/w<k>.pt + latents_meta.json (md5 per window, the capture
    format, so dswap_lib.load_window reads it).  If the dir exists, the stored latents are used (md5-checked).
(B) every listed decoder decodes the SAME stored latents (dswap_lib.decode_clip, keep rule) ->
    /mnt/ssd_data/vae_20261005/decoder_swap/headroom_dev/<clip>__<dec>.mkv (single eye 576x1024, FFV1) + .md5 + .json
    (decode seconds).  Names "cd@<seed>" decode only frames 0,4,8,... with that seed -> <clip>__cd@<seed>.npy (HR-SEED).
Existing outputs are never overwritten (skipped).
usage: CUDA_VISIBLE_DEVICES=0 flock /tmp/claude-gpu0.lock python headroom_v1.py <dec[,dec...]> <clip[,clip...]>
"""
import hashlib
import importlib.util
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dswap_lib as DL  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "infer_lossless", f"{DL.REPO}/scripts/distill/runs/fulldata_v2/beyond4/infer_lossless.py")
_IL = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_IL)
ROOT = "/mnt/ssd_data/vae_20261005/decoder_swap"
GTL, HR = f"{ROOT}/gt_lat_dev", f"{ROOT}/headroom_dev"
DECS, CLIPS = sys.argv[1].split(","), sys.argv[2].split(",")
T0 = time.time()


def log(*a):
    print(f"[headroom {time.time() - T0:7.1f}s]", *a, flush=True)


def gt(clip):
    return np.load(f"/mnt/ssd_data/deep_20261004/decoder_ft/gt_dev/{clip}_TR.npy", mmap_mode="r")


def encode_clip(clip):
    od = f"{GTL}/{clip}"
    if os.path.exists(f"{od}/latents_meta.json"):
        log(f"{clip}: GT latents exist, reusing")
        return
    enc = DL.Decoder("stock")             # the deployed VAE object (encoder + decoder), bf16
    X = gt(clip)
    n = X.shape[0]
    os.makedirs(od)
    meta = dict(clip=clip, src=f"/mnt/ssd_data/deep_20261004/decoder_ft/gt_dev/{clip}_TR.npy", n=n, windows=[])
    with torch.no_grad():
        for k, (i, cur_i, cur_ov) in enumerate(DL.windows(n)):
            x = torch.from_numpy(np.ascontiguousarray(X[cur_i:cur_i + DL.FC])).permute(0, 3, 1, 2).float() / 255.0
            xp = enc.ip.preprocess(x, height=x.shape[2], width=x.shape[3])
            z = DL.II._Pipe._encode_vae_frames(enc.shim, xp, torch.device("cuda"), 1, False, n_frames_per_time=5)
            lat = (z.float() * enc.vae.config.scaling_factor).to(torch.bfloat16).cpu().contiguous()
            assert tuple(lat.shape) == (1, DL.win_len(n, cur_i), 4, 72, 128), lat.shape
            p = f"{od}/w{k:03d}.pt"
            torch.save(lat, p)
            meta["windows"].append(dict(k=k, path=p, shape=list(lat.shape), dtype=str(lat.dtype),
                                        md5=hashlib.md5(lat.view(torch.int16).numpy().tobytes()).hexdigest(),
                                        cur_i=cur_i, cur_ov=cur_ov))
    meta["n_windows"] = len(meta["windows"])
    json.dump(meta, open(f"{od}/latents_meta.json", "w"), indent=1)
    enc.unload()
    log(f"{clip}: encoded {meta['n_windows']} windows of the real right eye (n={n})")


def main():
    os.makedirs(HR, exist_ok=True)
    for c in CLIPS:
        encode_clip(c)
    for name in DECS:
        base, seed = (name.split("@")[0], int(name.split("@")[1])) if "@" in name else (name, DL.SEED)
        todo = [c for c in CLIPS if not os.path.exists(f"{HR}/{c}__{name}.json")]
        if not todo:
            log(f"SKIP {name}: all outputs exist")
            continue
        dec = DL.Decoder(base, seed=seed)
        log(f"decoder {name} loaded: {dec.info()}")
        for c in todo:
            n = gt(c).shape[0]
            s0, f0 = dec.decode_seconds, dec.decoded_frames
            info = dict(clip=c, decoder=dec.info(), name=name, n=n, gt_latents=f"{GTL}/{c}")
            if "@" in name:
                frames = list(range(0, n, 4))
                arr = DL.decode_frames(dec, f"{GTL}/{c}", n, frames)
                p = f"{HR}/{c}__{name}.npy"
                np.save(p, arr)
                info.update(path=p, frames=frames)
            else:
                arr = DL.decode_clip(dec, f"{GTL}/{c}", n)
                p = f"{HR}/{c}__{name}.mkv"
                _IL._ffv1_write(np.ascontiguousarray(arr), 30.0, p)
                info.update(path=p)
            dsec, dfr = dec.decode_seconds - s0, dec.decoded_frames - f0
            info.update(md5=hashlib.md5(np.ascontiguousarray(arr).tobytes()).hexdigest(), decode_seconds=dsec,
                        decoded_frames=dfr, sec_per_decoded_frame=dsec / max(dfr, 1))
            json.dump(info, open(f"{HR}/{c}__{name}.json", "w"), indent=1)
            log(f"{c} {name}: md5 {info['md5']} decode {dsec:.1f}s / {dfr} frames")
        dec.unload()
        del dec
    log("HEADROOM_DECODE_DONE")


if __name__ == "__main__":
    main()
