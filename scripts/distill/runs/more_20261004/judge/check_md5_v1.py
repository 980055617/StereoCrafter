#!/usr/bin/env python
"""judge J2a (CPU, read-only): decode each render's FFV1 sbs with ffmpeg to rgb24 and md5 the whole stream (the same
method as finalcheck speed check_passthrough_v1.decode_md5); compare with its writer_md5.txt and an expected md5.
usage: check_md5_v1.py OUT.txt DIR=EXPECTED_MD5 [DIR=EXPECTED_MD5 ...]   (DIR may end in /rep1 for SK_REPEAT outputs)"""
import glob, hashlib, os, subprocess, sys
os.chdir("/home/kawa/master_project/StereoCrafter")
FFMPEG = "/home/kawa/miniconda3/envs/stereocrafter/bin/ffmpeg"
FFPROBE = "/home/kawa/miniconda3/envs/stereocrafter/bin/ffprobe"
def decode_md5(path):
    w, h = map(int, subprocess.check_output([FFPROBE, "-v", "error", "-select_streams", "v:0", "-show_entries",
               "stream=width,height", "-of", "csv=p=0", path]).decode().strip().split(",")[:2])
    fb = w * h * 3; p = subprocess.Popen([FFMPEG, "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                                        stdout=subprocess.PIPE)
    m, n = hashlib.md5(), 0
    while True:
        b = p.stdout.read(fb)
        if not b: break
        assert len(b) == fb; m.update(b); n += 1
    assert p.wait() == 0
    return m.hexdigest(), n, w, h
out, specs = sys.argv[1], sys.argv[2:]
L, nf = [], 0
for s in specs:
    d, exp = s.split("=")
    mk = glob.glob(f"{d}/*_sbs.mkv"); assert len(mk) == 1, (d, mk)
    wr = [ln.split()[0] for ln in open(f"{d}/writer_md5.txt") if "_sbs" in ln][0]
    got, n, w, h = decode_md5(mk[0])
    ok = (got == wr == exp); nf += (not ok)
    L.append(f"{'PASS' if ok else 'FAIL'} {d}: decoded {got} writer {wr} expected {exp} frames {n} {w}x{h}")
    print(L[-1], flush=True)
L.append(f"SUMMARY checked={len(specs)} failed={nf}")
open(out, "w").write("\n".join(L) + "\n"); print(L[-1])
