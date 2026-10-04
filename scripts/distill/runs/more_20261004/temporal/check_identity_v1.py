#!/usr/bin/env python
"""more_20261004 / temporal lane: integrity gates I2-I5 of PREREG.txt (CPU only; run with CUDA_VISIBLE_DEVICES="").

For every render dir outputs/more_20261004/temporal/clips/<clip>_<label>:
  I2  speed_log.json: schedule_windows == tlib.window_schedule(N, 14, overlap) and n_windows == len(schedule);
      for overlap 3 the window lengths equal BASE's (finalcheck speed <clip>_deliv_g100_T5nat speed_log num_frames).
  I3  RNG fingerprints vs BASE's speed_log init_md5 (natural draws):
        label contains pw / dcs / ctrl -> every window's init_md5 identical to BASE's
        label contains nsfa            -> every window's natural init_md5 identical; used_md5 identical in window 0 only
                                          (and noise_substituted == cur_overlap for windows >= 1)
        label contains ov5 / ov7       -> window 0 identical (later windows: identity count reported)
  I4  decoded right-eye frames (FFV1 -> rgb24, exact) vs BASE: index of the first differing frame; expected >= 14 for
      ov5 ov7 pw03 pw05 nsfa (window 0 = frames 0..13 identical); ctrl must be identical everywhere.
  I5  render log: deliverable -> "missing=1428" and "updated 5 gated modules"; origin -> "unet=None" and no
      "Partial UNet state load"; rc=0 (<dir>.rc).
usage: check_identity_v1.py OUT.txt DIR [DIR ...]
"""
import json
import os
import subprocess
import sys

import numpy as np

REPO = "/home/kawa/master_project/StereoCrafter"
os.chdir(REPO)
sys.path.insert(0, os.path.join(REPO, "scripts/distill/runs/more_20261004/temporal"))
from tlib import window_schedule  # noqa: E402

FFMPEG = "/home/kawa/miniconda3/envs/stereocrafter/bin/ffmpeg"
FFPROBE = "/home/kawa/miniconda3/envs/stereocrafter/bin/ffprobe"
BASE = "outputs/finalcheck_20261004/speed/clips/{c}_deliv_g100_T5nat"


def frames(path):
    out = subprocess.check_output([FFPROBE, "-v", "error", "-select_streams", "v:0", "-show_entries",
                                   "stream=width,height", "-of", "csv=p=0", path]).decode().strip()
    W, H = [int(x) for x in out.split(",")[:2]]
    p = subprocess.Popen([FFMPEG, "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         stdout=subprocess.PIPE)
    fb = W * H * 3
    while True:
        buf = p.stdout.read(fb)
        if not buf:
            break
        yield np.frombuffer(buf, np.uint8).reshape(H, W, 3)[:, W // 2:]
    assert p.wait() == 0


def first_diff(a_path, b_path):
    n = 0
    first = None
    ndiff = 0
    for fa, fb in zip(frames(a_path), frames(b_path)):
        if not np.array_equal(fa, fb):
            ndiff += 1
            if first is None:
                first = n
        n += 1
    return first, ndiff, n


def mkv(d):
    fs = [f for f in os.listdir(d) if f.endswith("_sbs.mkv")]
    assert len(fs) == 1, (d, fs)
    return os.path.join(d, fs[0])


out_txt = sys.argv[1]
L, nfail = [], 0
for d in sys.argv[2:]:
    d = d.rstrip("/")
    name = os.path.basename(d)
    clip, label = name.split("_", 1)
    base = BASE.format(c=clip)
    msgs, ok = [], True
    # I5 ----------------------------------------------------------------------------------------------
    rc = open(d + ".rc").read().split()[0] if os.path.exists(d + ".rc") else "?"
    log = open(d + ".log").read()
    is_origin = "unet=None" in log.split("[sk] clip=")[1].split("\n")[0] if "[sk] clip=" in log else False
    if is_origin:
        i5 = rc == "0" and "Partial UNet state load" not in log
    else:
        i5 = rc == "0" and "missing=1428" in log and "updated 5 gated modules" in log
    msgs.append(f"I5 {'ok' if i5 else 'FAIL'} (rc={rc} model={'origin' if is_origin else 'deliv'})")
    ok &= i5
    # I2 ----------------------------------------------------------------------------------------------
    S = json.load(open(os.path.join(d, "speed_log.json")))
    B = json.load(open(os.path.join(base, "speed_log.json")))
    sched = S["schedule_windows"]
    N = sched[-1]["keep_to"]
    i2 = (sched == window_schedule(N, 14, S["overlap"])) and S["n_windows"] == len(sched) \
        and [w["num_frames"] for w in S["windows"]] == [x["nf"] for x in sched]
    if S["overlap"] == 3:
        i2 &= [w["num_frames"] for w in S["windows"]] == [w["num_frames"] for w in B["windows"]]
    msgs.append(f"I2 {'ok' if i2 else 'FAIL'} (N={N} overlap={S['overlap']} windows={len(sched)} "
                f"lengths={sorted(set(x['nf'] for x in sched))} last={sched[-1]['nf']})")
    ok &= i2
    # I3 ----------------------------------------------------------------------------------------------
    a, b = S["init_md5"], B["init_md5"]
    same = [x == y for x, y in zip(a, b)]
    if is_origin:
        i3 = True
        msgs.append(f"I3 n/a (origin) init identical to BASE windows {sum(same)}/{len(a)}")
    elif "nsfa" in label:
        used_same = [x == y for x, y in zip(S["used_md5"], b)]
        sub_ok = S["noise_substituted"][0] == 0 and all(
            s == w["cur_overlap"] for s, w in zip(S["noise_substituted"][1:], sched[1:]))
        i3 = len(a) == len(b) and all(same) and used_same[0] and not any(used_same[1:]) and sub_ok
        msgs.append(f"I3 {'ok' if i3 else 'FAIL'} natural-draw identical {sum(same)}/{len(b)}, used identical "
                    f"{sum(used_same)} (window0 {used_same[0]}), substituted {S['noise_substituted']}")
    elif any(k in label for k in ("pw", "dcs", "ctrl")):
        i3 = len(a) == len(b) and all(same) and S["used_md5"] == a
        msgs.append(f"I3 {'ok' if i3 else 'FAIL'} every window init identical to BASE ({sum(same)}/{len(b)})")
    elif any(k in label for k in ("ov5", "ov7")):
        i3 = same[0]
        msgs.append(f"I3 {'ok' if i3 else 'FAIL'} window0 identical={same[0]}; same-index windows identical "
                    f"{sum(same)}/{min(len(a), len(b))} (windows {len(a)} vs BASE {len(b)})")
    else:
        i3 = True
        msgs.append(f"I3 n/a identical windows {sum(same)}/{min(len(a), len(b))}")
    ok &= i3
    # I4 ----------------------------------------------------------------------------------------------
    if is_origin:
        msgs.append("I4 n/a (origin)")
    else:
        first, ndiff, n = first_diff(mkv(d), mkv(base))
        if "ctrl" in label:
            i4 = first is None
        elif any(k in label for k in ("ov5", "ov7", "pw03", "pw05", "nsfa")):
            i4 = first is not None and first >= 14
        else:
            i4 = True
        msgs.append(f"I4 {'ok' if i4 else 'FAIL'} right eye vs BASE: first differing frame {first}, "
                    f"{ndiff}/{n} frames differ")
        ok &= i4
    nfail += (not ok)
    L.append(f"{'PASS' if ok else 'FAIL'} {d} :: " + " | ".join(msgs))
    print(L[-1], flush=True)
with open(out_txt, "a") as fh:
    fh.write("\n".join(L) + "\n" + f"SUMMARY checked={len(sys.argv) - 2} failed={nfail}\n")
print(f"SUMMARY checked={len(sys.argv) - 2} failed={nfail}")
