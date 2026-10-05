#!/usr/bin/env python
"""deep_20261004 / decoder_ft lane -- md5 GATES C1 / C2 / C3 (PREREG.txt section 3).  CPU, reads md5 sidecars only.

C1  capture render (capture/clips/<clip>_<lat>) writer md5 == the published lossless render's md5 (where one exists), and
    (PREREG_ADDENDUM_3) every FULL 14-frame window's initial-noise md5 == the common reference sequence (taken from the
    0082 origin capture, whose 15 windows are all full); the original "init_md5_all == 6db474cb.." rule only fits
    14-window clips and is reported alongside for reference
C2  <redec_root>/<clip>_<lat>__stock md5 == capture md5
C3  <redec_root>/<clip>_<lat>__<step0 name> md5 == capture md5 (when such a dir exists)
usage: python check_md5_v1.py <out_txt> <redec_root> <clip,...> [step0_name]
"""
import hashlib
import json
import os
import sys

REPO = "/home/kawa/master_project/StereoCrafter"
os.chdir(REPO)
CAP = "/mnt/ssd_data/deep_20261004/decoder_ft/capture/clips"
INIT = "6db474cbe7ebf98270a6d5399dd57987"
OUT, RED, CLIPS = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
S0 = sys.argv[4] if len(sys.argv) > 4 else None
assert not os.path.exists(OUT), OUT
PUB = {  # published lossless renders (md5 sidecars)
    "origin_cap": ["outputs/beyond4_lossless/clips/{c}_origin_ll", "outputs/deep_20261004/scale_gt/renders/{c}_origin_s8"],
    "deliv_cap": ["outputs/beyond_distil_mamba_scaled/clips/{c}_mstudent2_step800_deliv_ll",
                  "outputs/beyond_distil_mamba_scaled/clips/{c}_mstudent2_step800_ll"],   # same weights (md5-identical renders)
}


def md5_of(d):
    p = [f for f in os.listdir(d) if f.endswith("_sbs.mkv.md5")] if os.path.isdir(d) else []
    return open(os.path.join(d, p[0])).read().split()[0] if p else None


_r = json.load(open(f"{CAP}/0082_origin_cap/speed_log.json"))
assert all(w["num_frames"] == 14 for w in _r["windows"])
REF = _r["init_md5"]
lines, ok = [], True
for c in CLIPS:
    for lat in ("origin_cap", "deliv_cap"):
        cd = f"{CAP}/{c}_{lat}"
        cm = md5_of(cd)
        sl = json.load(open(f"{cd}/speed_log.json"))
        init = hashlib.md5("".join(sl["init_md5"]).encode()).hexdigest()
        full_ok = all(m == REF[k] for k, (m, w) in enumerate(zip(sl["init_md5"], sl["windows"])) if w["num_frames"] == 14)
        pubs = [(pd.format(c=c), md5_of(pd.format(c=c))) for pd in PUB[lat]]
        pubs = [(p, m) for p, m in pubs if m is not None]
        c1 = full_ok and all(m == cm for _, m in pubs)
        st = md5_of(f"{RED}/{c}_{lat}__stock")
        c2 = st == cm
        msg = (f"{c} {lat:10s} capture {cm} full-window noise {'OK' if full_ok else 'BAD'} (14-window fingerprint "
               f"{'match' if init == INIT else 'n/a: ' + str(len(sl['init_md5'])) + ' windows'}); published "
               f"{[(os.path.basename(p), m == cm) for p, m in pubs] or 'none (dev deliverable: capture is the reference)'}"
               f" -> C1 {'PASS' if c1 else 'FAIL'}; stock re-decode {st} -> C2 {'PASS' if c2 else 'FAIL'}")
        ok &= c1 and c2
        if S0 and os.path.isdir(f"{RED}/{c}_{lat}__{S0}"):
            s0 = md5_of(f"{RED}/{c}_{lat}__{S0}")
            c3 = s0 == cm
            ok &= c3
            msg += f"; {S0} re-decode {s0} -> C3 {'PASS' if c3 else 'FAIL'}"
        lines.append(msg)
lines.append(f"GATES {'ALL PASS' if ok else 'FAILURE'}")
open(OUT, "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
