#!/usr/bin/env python
"""G6 pass-through + lossless round-trip check (CPU only; run with CUDA_VISIBLE_DEVICES="").

For every render dir given: decode <dir>/*_sbs.mkv with ffmpeg to rgb24 (FFV1 is exact), then
  - full md5 of the decoded (T,H,W,3) uint8 array  == the writer md5 recorded in <dir>/writer_md5.txt (sbs line)
  - left-half md5 of [:, :, :W/2]                  == origin@1.01's decoded left half for the same clip
  - frame count                                    == origin@1.01's frame count
Reference left halves come from outputs/beyond4_lossless/clips/<clip>_origin_ll/*_sbs.mkv (decoded the same way);
they are cached in the JSON given as --cache.

usage: check_passthrough_v1.py OUT.txt CACHE.json DIR [DIR ...]
"""
import glob
import hashlib
import json
import os
import subprocess
import sys

import numpy as np

REPO = "/home/kawa/master_project/StereoCrafter"
FFMPEG = "/home/kawa/miniconda3/envs/stereocrafter/bin/ffmpeg"
FFPROBE = "/home/kawa/miniconda3/envs/stereocrafter/bin/ffprobe"
os.chdir(REPO)


def dims(path):
    out = subprocess.check_output([FFPROBE, "-v", "error", "-select_streams", "v:0", "-show_entries",
                                   "stream=width,height", "-of", "csv=p=0", path]).decode().strip()
    w, h = out.split(",")[:2]
    return int(w), int(h)


def decode_md5(path):
    W, H = dims(path)
    fb = W * H * 3
    p = subprocess.Popen([FFMPEG, "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         stdout=subprocess.PIPE)
    full, left, n = hashlib.md5(), hashlib.md5(), 0
    while True:
        buf = p.stdout.read(fb)
        if not buf:
            break
        assert len(buf) == fb, f"short frame in {path}"
        full.update(buf)
        fr = np.frombuffer(buf, np.uint8).reshape(H, W, 3)
        left.update(np.ascontiguousarray(fr[:, : W // 2]).tobytes())
        n += 1
    assert p.wait() == 0, f"ffmpeg decode failed {path}"
    return dict(full=full.hexdigest(), left=left.hexdigest(), n=n, W=W, H=H)


def writer_md5(d):
    for line in open(os.path.join(d, "writer_md5.txt")):
        if "_sbs" in line:
            return line.split()[0]
    return None


out_txt, cache_json = sys.argv[1], sys.argv[2]
dirs = sys.argv[3:]
cache = json.load(open(cache_json)) if os.path.exists(cache_json) else {}
lines, nfail = [], 0
for d in dirs:
    d = d.rstrip("/")
    clip = os.path.basename(d).split("_")[0]
    if clip not in cache:
        ref = glob.glob(f"outputs/beyond4_lossless/clips/{clip}_origin_ll/*_sbs.mkv")
        assert len(ref) == 1, ref
        r = decode_md5(ref[0])
        r["writer"] = writer_md5(os.path.dirname(ref[0]))
        r["roundtrip_ok"] = r["full"] == r["writer"]
        cache[clip] = r
        json.dump(cache, open(cache_json, "w"), indent=1)
    ref = cache[clip]
    mk = glob.glob(f"{d}/*_sbs.mkv")
    if len(mk) != 1:
        lines.append(f"FAIL {d}: no single sbs.mkv ({mk})")
        nfail += 1
        continue
    r = decode_md5(mk[0])
    wm = writer_md5(d)
    rt = r["full"] == wm
    lp = (r["left"] == ref["left"]) and (r["n"] == ref["n"])
    ok = rt and lp and ref["roundtrip_ok"]
    nfail += (not ok)
    lines.append(f"{'PASS' if ok else 'FAIL'} {d}  frames={r['n']} (ref {ref['n']})  roundtrip(full decode md5 == writer md5)={rt} "
                 f"[{r['full']}]  left==origin@1.01 left={lp} [{r['left']}]  ref_roundtrip={ref['roundtrip_ok']}")
with open(out_txt, "a") as fh:
    for ln in lines:
        fh.write(ln + "\n")
        print(ln)
    fh.write(f"SUMMARY checked={len(dirs)} failed={nfail}\n")
print(f"SUMMARY checked={len(dirs)} failed={nfail}")
