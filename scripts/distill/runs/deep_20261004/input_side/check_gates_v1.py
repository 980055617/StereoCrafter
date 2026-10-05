#!/usr/bin/env python
"""input_side lane gates G1/G3/G4 (CPU; run with CUDA_VISIBLE_DEVICES="").
G1  renders whose label starts with 'g1_dep_' : writer md5 == the existing render's writer md5 (same clip, model)
G3  per-window initial-latent md5 list (speed_log.json init_md5) identical for every render of a clip
G4  FFV1 decode md5 == writer md5; decoded LEFT half md5 + frame count == the existing origin_ll render's decoded left
    (decode_md5 copied from scripts/distill/runs/finalcheck_20261004/speed/check_passthrough_v1.py)
usage: check_gates_v1.py OUT.txt DIR [DIR ...]      (DIR = outputs/deep_20261004/input_side/clips/<clip>_<label>)
"""
import hashlib
import json
import os
import subprocess
import sys
from collections import defaultdict

import numpy as np

REPO = "/home/kawa/master_project/StereoCrafter"
FFMPEG = "/home/kawa/miniconda3/envs/stereocrafter/bin/ffmpeg"
FFPROBE = "/home/kawa/miniconda3/envs/stereocrafter/bin/ffprobe"
os.chdir(REPO)
EXIST = {"origin": "outputs/beyond4_lossless/clips/{c}_origin_ll",
         "deliv": "outputs/beyond_distil_mamba_scaled/clips/{c}_mstudent2_step800_deliv_ll"}


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


def sbs(d):
    c = [f for f in os.listdir(d) if f.endswith("_sbs.mkv")]
    assert len(c) == 1, (d, c)
    return os.path.join(d, c[0])


OUT = sys.argv[1]
assert not os.path.exists(OUT), f"refusing to overwrite {OUT}"
dirs = sys.argv[2:]
L, nfail = [], 0
refleft = {}
fps = defaultdict(dict)
for d in dirs:
    base = os.path.basename(d.rstrip("/"))
    clip, label = base.split("_", 1)
    model = "deliv" if label.endswith("deliv") else "origin"
    wm = writer_md5(d)
    dec = decode_md5(sbs(d))
    if clip not in refleft:
        refleft[clip] = decode_md5(sbs(EXIST["origin"].format(c=clip)))
    ok4 = dec["full"] == wm and dec["left"] == refleft[clip]["left"] and dec["n"] == refleft[clip]["n"]
    nfail += (not ok4)
    L.append(f"G4 {'PASS' if ok4 else 'FAIL'} {base}: decode==writer {dec['full'] == wm}  left==origin_ll left "
             f"{dec['left'] == refleft[clip]['left']}  frames {dec['n']} (ref {refleft[clip]['n']})")
    if label.startswith("g1_dep_"):
        ref = writer_md5(EXIST[model].format(c=clip))
        ok1 = wm == ref
        nfail += (not ok1)
        L.append(f"G1 {'PASS' if ok1 else 'FAIL'} {base}: writer md5 {wm} vs existing {model} {ref}")
    sp = os.path.join(d, "speed_log.json")
    fps[clip][base] = json.load(open(sp))["init_md5"]
for clip, recs in fps.items():
    uniq = {tuple(v) for v in recs.values()}
    ok3 = len(uniq) == 1
    nfail += (not ok3)
    L.append(f"G3 {'PASS' if ok3 else 'FAIL'} {clip}: {len(recs)} renders, {len(next(iter(uniq)))} windows, "
             f"{len(uniq)} distinct init-latent fingerprint lists")
L.append(f"TOTAL failures: {nfail}")
open(OUT, "w").write("\n".join(L) + "\n")
print("\n".join(L))
