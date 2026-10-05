#!/usr/bin/env python
"""vae_20261005 / decoder_swap -- RE-DECODE captured pre-decode latents with each decoder (PREREG.txt sections 1 and 3).

Latents: decoder_ft captures /mnt/ssd_data/deep_20261004/decoder_ft/latents/<clip>_<latlabel>/w<k>.pt (read only, md5-checked
per window against latents_meta.json).  Decoding: dswap_lib.decode_clip (keep rule, redecode_v1.py's uint8 path).
SBS: left half = the capture render's left half (decoder_ft/capture/clips, lossless FFV1, identical by construction),
right half = the decoded frames; FFV1 via beyond4/infer_lossless.py _ffv1_write (+ .md5 of the pre-encode array,
writer_md5.txt, redecode.json with decode seconds).  An existing output dir is never overwritten (skipped).
usage: CUDA_VISIBLE_DEVICES=0 flock /tmp/claude-gpu0.lock python redecode_swap_v1.py <out_root> <dec[,dec...]> <clip:latlabel>[,...]
"""
import hashlib
import importlib.util
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dswap_lib as DL  # noqa: E402  (chdir REPO)
from decord import VideoReader, cpu  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "infer_lossless", f"{DL.REPO}/scripts/distill/runs/fulldata_v2/beyond4/infer_lossless.py")
_IL = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_IL)
LATR = "/mnt/ssd_data/deep_20261004/decoder_ft/latents"
CAPR = "/mnt/ssd_data/deep_20261004/decoder_ft/capture/clips"
T0 = time.time()


def log(*a):
    print(f"[redecode_swap {time.time() - T0:7.1f}s]", *a, flush=True)


def left_half(clip, latlabel, n):
    p = f"{CAPR}/{clip}_{latlabel}/{clip}_inpainting_results_sbs.mkv"
    vr = VideoReader(p, ctx=cpu(0))
    assert len(vr) == n, (p, len(vr), n)
    a = vr.get_batch(list(range(n))).asnumpy()
    return np.ascontiguousarray(a[:, :, : a.shape[2] // 2]), p


def main():
    out_root, decs, pairs = sys.argv[1], sys.argv[2].split(","), [x.split(":") for x in sys.argv[3].split(",")]
    os.makedirs(out_root, exist_ok=True)
    lefts = {}
    for name in decs:
        todo = [(c, l) for c, l in pairs if not os.path.exists(f"{out_root}/{c}_{l}__{name}")]
        for c, l in pairs:
            if (c, l) not in todo:
                log(f"SKIP {out_root}/{c}_{l}__{name} exists")
        if not todo:
            continue
        dec = DL.Decoder(name)
        log(f"decoder {name} loaded: {dec.info()}")
        for clip, latlabel in todo:
            od = f"{out_root}/{clip}_{latlabel}__{name}"
            vr = VideoReader(f"video_data/splatting/{clip}_splatting_results.mp4", ctx=cpu(0))
            n, fps = len(vr), float(vr.get_avg_fps())
            del vr
            s0, f0 = dec.decode_seconds, dec.decoded_frames
            t0 = time.time()
            right = DL.decode_clip(dec, f"{LATR}/{clip}_{latlabel}", n)
            wall = time.time() - t0
            dsec, dfr = dec.decode_seconds - s0, dec.decoded_frames - f0
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
            nw = len(DL.windows(n))
            json.dump(dict(clip=clip, latlabel=latlabel, decoder=dec.info(), left_from=lp, n=n, fps=fps, md5=dig,
                           right_md5=hashlib.md5(np.ascontiguousarray(right).tobytes()).hexdigest(),
                           decode_seconds=dsec, decoded_frames=dfr, sec_per_decoded_frame=dsec / max(dfr, 1),
                           n_windows=nw, deployed_structure_seconds=(dsec / max(dfr, 1)) * DL.FC * nw,
                           wall_seconds_incl_postprocess=wall, gpu=os.environ.get("CUDA_VISIBLE_DEVICES")),
                      open(f"{od}/redecode.json", "w"), indent=1)
            log(f"wrote {p} md5={dig} n={n} decode {dsec:.1f}s for {dfr} frames ({dsec / max(dfr, 1):.3f} s/frame), wall {wall:.1f}s")
        dec.unload()
        del dec
    log("REDECODE_SWAP_DONE")


if __name__ == "__main__":
    main()
