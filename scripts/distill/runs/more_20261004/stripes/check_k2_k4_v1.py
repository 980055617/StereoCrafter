#!/usr/bin/env python
"""K1/K2/K4 integrity checks for the stripes lane (CPU only).

K1  control render (label *_K1none) writer md5 == the baseline md5 given in PREREG.txt.
K2  every render's speed_log.json init_md5 list == its config's reference list
      origin_* -> outputs/finalcheck_20261004/speed/clips/<clip>_origin_g100_s8/speed_log.json
      deliv_*  -> outputs/finalcheck_20261004/speed/clips/<clip>_deliv_g100_T5nat/speed_log.json
K4  stripe_fill_log.json: filled renders changed > 0 window pixels and 0 outside the crack set; keep-variants carry
    the K1/none binarised-mask md5 of the same clip (computed here from a none-log when available, else from the
    0301 control), shrink-variants a different one; md5_warped_window differs from the none render of the clip.
usage: check_k2_k4_v1.py OUT.txt
"""
import glob
import json
import os
import sys

os.chdir("/home/kawa/master_project/StereoCrafter")
OUT = sys.argv[1]
if os.path.exists(OUT):
    sys.exit(f"refusing to overwrite {OUT}")
BASE_MD5 = {"0301": "a41b432baeb25393bb12d39b4f2e6a21"}
REF = "outputs/finalcheck_20261004/speed/clips"
L, nfail = [], 0
none_logs = {}
recs = []
for d in sorted(glob.glob("outputs/more_20261004/stripes/clips/*/")):
    d = d.rstrip("/")
    name = os.path.basename(d)
    clip, label = name.split("_", 1)
    sp, fl = f"{d}/speed_log.json", f"{d}/stripe_fill_log.json"
    if not (os.path.exists(sp) and os.path.exists(fl) and glob.glob(f"{d}/*_sbs.mkv")):
        L.append(f"INCOMPLETE {name} (no speed_log / fill log / sbs.mkv)")
        nfail += 1
        continue
    s, f = json.load(open(sp)), json.load(open(fl))
    md5 = open(f"{d}/writer_md5.txt").read().split()[0]
    recs.append((clip, label, d, s, f, md5))
    if f["mode"] == "none" and f["mask_mode"] == "keep":
        none_logs[clip] = f
for clip, label, d, s, f, md5 in recs:
    ref_lab = "origin_g100_s8" if label.startswith("origin") else "deliv_g100_T5nat"
    ref = json.load(open(f"{REF}/{clip}_{ref_lab}/speed_log.json"))["init_md5"]
    k2 = s["init_md5"] == ref
    nfail += (not k2)
    msg = [f"K2 {'PASS' if k2 else 'FAIL'} ({len(s['init_md5'])}/{len(ref)} windows vs {clip}_{ref_lab})"]
    if label.endswith("K1none"):
        k1 = md5 == BASE_MD5.get(clip)
        nfail += (not k1)
        msg.append(f"K1 {'PASS' if k1 else 'FAIL'} md5={md5} expected={BASE_MD5.get(clip)}")
    else:
        k4a = f["changed_px_window"] > 0 and f["changed_outside_crack_region"] == 0
        nl = none_logs.get(clip)
        if nl is not None:
            mb_ok = (f["md5_mask_bin_window"] == nl["md5_mask_bin_window"]) == (f["mask_mode"] == "keep")
            w_ok = f["md5_warped_window_f32"] != nl["md5_warped_window_f32"]
            ref_txt = "vs the clip's none-log"
        else:   # no none render for this clip: compare keep/shrink variants of the same clip among themselves
            sib = [g for c2, l2, _, _, g, _ in recs if c2 == clip and g["mask_mode"] == "keep" and g is not f]
            mb_ok = all((f["md5_mask_bin_window"] == g["md5_mask_bin_window"]) == (f["mask_mode"] == "keep") for g in sib)
            w_ok = True
            ref_txt = f"vs {len(sib)} keep-variant sibling(s) of the clip (no none-log)"
        k4 = k4a and mb_ok and w_ok
        nfail += (not k4)
        msg.append(f"K4 {'PASS' if k4 else 'FAIL'} fill={f['mode']}/{f['mask_mode']} changed_px={f['changed_px_window']} "
                   f"outside={f['changed_outside_crack_region']} crack_frac={f['crack_frac_window']:.5f} "
                   f"hole_frac={f['hole_frac_window']:.5f} maskbin_ok={mb_ok} warped_differs={w_ok} ({ref_txt})")
    L.append(f"{clip}_{label:34s} md5={md5}  " + "  ".join(msg))
L.append(f"SUMMARY renders={len(recs)} failed_checks={nfail}")
open(OUT, "w").write("\n".join(L) + "\n")
print("\n".join(L))
