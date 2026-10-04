#!/usr/bin/env python
"""ays_20261004 / verify -- V3 pairing confirmation on 0204 and 0259 (CPU only; own code; read-only).  PREREG.txt V3.

(a) per-window init_md5 (speed_log.json windows[k].init_md5) identical, window for window and in number, across every
    RNG-paired render that has a speed_log; the unpadded T5nat render must match at window 0 only.
(b) config per render: guid, sigmas, rng_pad_to, calls/window, batch sizes, pad draws, unet_state.
(c) ffmpeg rgb24 decode of every render read for the clip: full-array md5 == writer md5; left-half md5 identical across
    all renders; equal frame counts.
(d) 0204 only: origin_ll md5 == finalcheck 0204_origin_g101_s8 md5; deliverable 8x2 md5 == finalcheck 0204_deliv_g101_s8.
usage: python check_pairing_verify_v1.py OUT.txt     exit 0 iff every check passes; refuses to overwrite OUT.txt
"""
import hashlib
import json
import os
import subprocess
import sys

import numpy as np

os.chdir("/home/kawa/master_project/StereoCrafter")
OUTF = sys.argv[1]
assert not os.path.exists(OUTF), f"refusing to overwrite {OUTF}"
FFMPEG = "/home/kawa/miniconda3/envs/stereocrafter/bin/ffmpeg"
FFPROBE = "/home/kawa/miniconda3/envs/stereocrafter/bin/ffprobe"
DELIV = "/mnt/ssd_data/stereocrafter_weights/_distill_injected/mamba5slot_plus_stepdistil_up3_train10clip_step800_20261001.pt"
AYS5 = [700.0, 11.25711441040039, 1.7890000343322754, 0.26404356956481934, 0.0020000000949949026]
AYS8 = [700.0, 32.1326904296875, 8.801949501037598, 3.318009376525879, 1.1647257804870605, 0.35714080929756165,
        0.06827998906373978, 0.0020000000949949026]
T5 = [700.0, 7.276163101196289, 1.1675708293914795, 0.09738767892122269, 0.0020000000949949026]
F = "outputs/finalcheck_20261004/speed/clips"
JA = "outputs/more_20261004/judge/ays/clips"
A5 = "outputs/ays_20261004/ays5/clips"

# name -> (dir, expected config or None (no speed_log), pairing class)
#   class "all" = every window identical to the reference; "w0" = window 0 only; None = no speed_log
#   config = (guid, sigmas, rng_pad_to, calls, batch, pad_per_window, unet_state)
def renders(c):
    r = {
        "origin_ll": (f"outputs/beyond4_lossless/clips/{c}_origin_ll", None, None),
        "mstudent2_step800_deliv_ll": (f"outputs/beyond_distil_mamba_scaled/clips/{c}_mstudent2_step800_deliv_ll", None, None),
        "s25_ll": (f"outputs/skeptic1_stack/clips/{c}_s25_ll", None, None),
        "AYS8_origin_g101": (f"{JA}/{c}_AYS8_origin_g101", (1.01, AYS8, None, 8, 2, 0, None), "all"),
        "AYS8_origin_g100": (f"{A5}/{c}_AYS8_origin_g100" + ("_r1" if c == "0204" else ""), (1.0, AYS8, None, 8, 1, 0, None), "all"),
        "deliv_g100_T5pad": (f"{F}/{c}_deliv_g100_T5pad", (1.0, T5, 8, 5, 1, 3, DELIV), "all"),
        "origin_g100_T5pad": (f"{F}/{c}_origin_g100_T5pad", (1.0, T5, 8, 5, 1, 3, None), "all"),
        "deliv_g100_T5nat": (f"{F}/{c}_deliv_g100_T5nat", (1.0, T5, None, 5, 1, 0, DELIV), "w0"),
        "origin_g100_s8": (f"{F}/{c}_origin_g100_s8", (1.0, None, None, 8, 1, 0, None), "all"),
        "deliv_g100_s8": (f"{F}/{c}_deliv_g100_s8", (1.0, None, None, 8, 1, 0, DELIV), "all"),
    }
    if c == "0204":
        r["AYS5pad8_origin_g100"] = (f"{JA}/{c}_AYS5pad8_origin_g100", (1.0, AYS5, 8, 5, 1, 3, None), "all")
        r["AYS5pad8_origin_g100__C0dup"] = (f"{A5}/{c}_AYS5pad8_origin_g100", (1.0, AYS5, 8, 5, 1, 3, None), "all")
        r["AYS5pad8_deliv_g100"] = (f"{A5}/{c}_AYS5pad8_deliv_g100", (1.0, AYS5, 8, 5, 1, 3, DELIV), "all")
        r["origin_g101_s8"] = (f"{F}/{c}_origin_g101_s8", (1.01, None, None, 8, 2, 0, None), "all")
        r["deliv_g101_s8"] = (f"{F}/{c}_deliv_g101_s8", (1.01, None, None, 8, 2, 0, DELIV), "all")
    else:
        r["AYS5pad8_origin_g100"] = (f"{A5}/{c}_AYS5pad8_origin_g100", (1.0, AYS5, 8, 5, 1, 3, None), "all")
    return r


def mkv(d):
    c = os.path.basename(d).split("_")[0]
    return f"{d}/{c}_inpainting_results_sbs.mkv"


def writer_md5(d):
    w = [ln.split()[0] for ln in open(f"{d}/writer_md5.txt") if "_sbs" in ln]
    assert len(w) == 1, (d, w)
    return w[0]


def decode(path):
    out = subprocess.check_output([FFPROBE, "-v", "error", "-select_streams", "v:0", "-show_entries",
                                   "stream=width,height", "-of", "csv=p=0", path]).decode().strip()
    W, H = [int(x) for x in out.split(",")[:2]]
    fb = W * H * 3
    p = subprocess.Popen([FFMPEG, "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         stdout=subprocess.PIPE)
    full, left, n = hashlib.md5(), hashlib.md5(), 0
    while True:
        buf = p.stdout.read(fb)
        if not buf:
            break
        assert len(buf) == fb, f"short frame {path}"
        full.update(buf)
        left.update(np.ascontiguousarray(np.frombuffer(buf, np.uint8).reshape(H, W, 3)[:, : W // 2]).tobytes())
        n += 1
    assert p.wait() == 0, f"ffmpeg failed {path}"
    return full.hexdigest(), left.hexdigest(), n, (H, W)


lines, nfail = [], 0


def rep(ok, s):
    global nfail
    nfail += (not ok)
    lines.append(("PASS " if ok else "FAIL ") + s)
    print(lines[-1], flush=True)


for c in ("0204", "0259"):
    lines.append(f"===== {c}")
    R = renders(c)
    ref_name = "deliv_g100_T5pad"
    ref_init = [w["init_md5"] for w in json.load(open(f"{R[ref_name][0]}/speed_log.json"))["windows"]]
    lines.append(f"reference {ref_name}: n_windows {len(ref_init)}; window-0 init_md5 {ref_init[0]}")
    # (a) + (b)
    for name, (d, cfg, cls) in R.items():
        if cfg is None:
            continue
        j = json.load(open(f"{d}/speed_log.json"))
        init = [w["init_md5"] for w in j["windows"]]
        if cls == "all":
            same = sum(a == b for a, b in zip(init, ref_init))
            ok_a = same == len(init) == len(ref_init)
            sa = f"init_md5 {same}/{len(init)} identical to {ref_name} (all windows required)"
        else:
            same = sum(a == b for a, b in zip(init, ref_init))
            ok_a = init[0] == ref_init[0] and len(init) == len(ref_init)
            sa = f"init_md5 window0 {'identical' if init[0] == ref_init[0] else 'DIFFERENT'}; {same}/{len(init)} identical overall (expected: window 0 only)"
        guid, sig, pad, calls, batch, padw, unet = cfg
        errs = []
        if j.get("guid") != guid: errs.append(f"guid {j.get('guid')} != {guid}")
        if j.get("sigmas") != sig: errs.append(f"sigmas {j.get('sigmas')} != {sig}")
        if j.get("rng_pad_to") != pad: errs.append(f"rng_pad_to {j.get('rng_pad_to')} != {pad}")
        if j.get("unet_calls_per_window") != [calls]: errs.append(f"calls {j.get('unet_calls_per_window')} != [{calls}]")
        if j.get("batch_sizes") != [batch]: errs.append(f"batch {j.get('batch_sizes')} != [{batch}]")
        if j.get("pad_draws_total") != padw * j["n_windows"]: errs.append(f"pad_draws {j.get('pad_draws_total')} != {padw}x{j['n_windows']}")
        if j.get("unet_state") != unet: errs.append(f"unet_state {j.get('unet_state')} != {unet}")
        if sig is not None:
            want = [repr(float(s)) for s in sig] + ["0.0"]
            sch = j.get("schedule") or {}
            if not sch or any(v["sigmas"] != want for v in sch.values()):
                errs.append("installed schedule sigmas differ")
        rep(ok_a and not errs, f"{c} {name}: {sa}; guid {j.get('guid')} calls {j.get('unet_calls_per_window')} batch "
            f"{j.get('batch_sizes')} pad_draws {j.get('pad_draws_total')} unet {'deliverable' if j.get('unet_state') == DELIV else j.get('unet_state')}"
            + ("" if not errs else "  ERRORS: " + "; ".join(errs)) + f"  [{d}]")
    # (c)
    lefts, counts, fulls = {}, {}, {}
    for name, (d, cfg, cls) in R.items():
        full, left, n, hw = decode(mkv(d))
        wm = writer_md5(d)
        fulls[name], lefts[name], counts[name] = full, left, n
        rep(full == wm, f"{c} {name}: decoded full md5 {full} {'==' if full == wm else '!='} writer {wm}; frames {n}; left {left[:12]}  [{mkv(d)}]")
    rep(len(set(lefts.values())) == 1 and len(set(counts.values())) == 1,
        f"{c} left halves: {len(set(lefts.values()))} distinct md5 over {len(lefts)} renders; frame counts {sorted(set(counts.values()))}")
    # (d)
    if c == "0204":
        rep(fulls["origin_ll"] == fulls["origin_g101_s8"],
            f"0204 origin_ll {fulls['origin_ll']} vs finalcheck origin_g101_s8 {fulls['origin_g101_s8']} (ties the 16-eval origin to the 8-step stream)")
        rep(fulls["mstudent2_step800_deliv_ll"] == fulls["deliv_g101_s8"],
            f"0204 mstudent2_step800_deliv_ll {fulls['mstudent2_step800_deliv_ll']} vs finalcheck deliv_g101_s8 {fulls['deliv_g101_s8']} (ties the 16-eval deliverable)")
        rep(fulls["AYS5pad8_origin_g100"] == fulls["AYS5pad8_origin_g100__C0dup"],
            f"0204 AYS5 origin judge (GPU 1) {fulls['AYS5pad8_origin_g100']} vs ays5 C0 re-render (GPU 0) {fulls['AYS5pad8_origin_g100__C0dup']}")
    else:
        lines.append("0259: no finalcheck 8-step @1.01 render exists; origin_ll / deliverable 8x2 are tied to the 8-step "
                     "stream by finalcheck PREREG G3/G7 (md5-verified on 0204 and 0301), not by a check on this clip")
lines.append(f"SUMMARY checks_failed={nfail}")
open(OUTF, "w").write("ays_20261004 / verify -- V3 pairing confirmation (PREREG.txt V3)\n" + "\n".join(lines) + "\n")
print(lines[-1])
sys.exit(1 if nfail else 0)
