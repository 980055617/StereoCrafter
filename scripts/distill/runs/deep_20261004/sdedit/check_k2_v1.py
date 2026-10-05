#!/usr/bin/env python
"""K2 RNG pairing + schedule check for every SDEdit render of this lane (CPU only).
For each outputs/deep_20261004/sdedit/clips/<clip>_<model>_g100_<v>/ with an sdedit_log.json:
  - per-window paired_ref_md5 == per-window init_md5 of the speed lane's <clip>_origin_g100_s8 AND <clip>_deliv_g100_s8
    (outputs/finalcheck_20261004/speed/clips/*/speed_log.json; the two references must agree with each other)
  - per-window UNet calls == 5 / 4 / 3 for sd31 / sd7 / sd1(fill); pad draws == 8 - calls; batch size 1
  - logged sigma_start == the expected grid value; schedule default8_index == the expected tail
  - warpfill: fill log present, changed_px_window > 0, map hit for every window (implied by a finished render)
usage: check_k2_v1.py OUT.txt [clip ...]     (appends; prints a SUMMARY line)
"""
import glob
import json
import os
import sys

os.chdir("/home/kawa/master_project/StereoCrafter")
OUT = sys.argv[1]
CLIPS = set(sys.argv[2:])
EXP = {"sd31": ("30.993608474731445", 5, [3, 4, 5, 6, 7]), "sd7": ("7.276163101196289", 4, [4, 5, 6, 7]),
       "sd1": ("1.1675708293914795", 3, [5, 6, 7]), "sd1fill": ("1.1675708293914795", 3, [5, 6, 7])}
L, nfail, nchk = [], 0, 0
for d in sorted(glob.glob("outputs/deep_20261004/sdedit/clips/*_g100_sd*")):
    if not os.path.isdir(d):
        continue
    base = os.path.basename(d)
    clip, model, _, v = base.split("_")
    if CLIPS and clip not in CLIPS:
        continue
    nchk += 1
    errs = []
    try:
        sd = json.load(open(f"{d}/sdedit_log.json"))
        sp = json.load(open(f"{d}/speed_log.json"))
    except Exception as e:  # noqa: BLE001
        L.append(f"FAIL {base}: logs missing ({e})")
        nfail += 1
        continue
    refs = {}
    for m in ("origin", "deliv"):
        p = f"outputs/finalcheck_20261004/speed/clips/{clip}_{m}_g100_s8/speed_log.json"
        refs[m] = json.load(open(p))["init_md5"]
    if refs["origin"] != refs["deliv"]:
        errs.append("the two g100_s8 references disagree")
    ref = refs["origin"]
    got = sd["paired_ref_md5"]
    same = sum(a == b for a, b in zip(got, ref))
    if len(got) != len(ref) or same != len(ref):
        errs.append(f"pairing {same}/{len(ref)} windows (n {len(got)})")
    s0, ncall, idx = EXP[v]
    if sd["mode"] != ("warpfill" if v == "sd1fill" else "warp"):
        errs.append(f"mode {sd['mode']}")
    if any(w["sigma_start"] != s0 for w in sd["windows"]):
        errs.append(f"sigma_start {sorted(set(w['sigma_start'] for w in sd['windows']))} != {s0}")
    calls = sorted(set(w["unet_calls"] for w in sp["windows"]))
    pads = sorted(set(w["pad_draws"] for w in sp["windows"]))
    bs = sp["batch_sizes"]
    if calls != [ncall] or pads != [8 - ncall] or bs != [1]:
        errs.append(f"calls {calls} pads {pads} batch {bs}")
    sch = list((sp.get("schedule") or {}).values())
    if not sch or sch[0]["default8_index"] != idx:
        errs.append(f"schedule index {sch[0]['default8_index'] if sch else None} != {idx}")
    fill = ""
    if v == "sd1fill":
        f = sd.get("fill") or {}
        if f and f.get("hole_px_window", -1) == 0:
            # PREREG_ADDENDUM_1 A3: no hole pixel in the window -> the fill is a no-op; must equal the sd1 render bit for bit
            def _wm(dd):
                for ln in open(f"{dd}/writer_md5.txt"):
                    if "_sbs" in ln:
                        return ln.split()[0]
            d_sd1 = d.replace("_g100_sd1fill", "_g100_sd1")
            lat0 = all(w.get("fill_latent_frac_changed", 1.0) == 0.0 for w in sd["windows"])
            if os.path.exists(f"{d_sd1}/writer_md5.txt"):
                m_fill, m_sd1 = _wm(d), _wm(d_sd1)
                if m_fill != m_sd1:
                    errs.append(f"zero-hole clip but writer md5 {m_fill} != sd1 {m_sd1}")
                fill = f" fill: ZERO-HOLE window (A3): writer md5 == sd1 render: {m_fill == m_sd1}; init latents unchanged: {lat0}"
            else:
                if not lat0:
                    errs.append("zero-hole clip but the fill changed the init latents")
                fill = f" fill: ZERO-HOLE window (A3): no sd1 render to compare; init latents unchanged by the fill: {lat0}"
        elif not f or f.get("changed_px_window", 0) <= 0:
            errs.append("fill log missing or nothing filled")
        else:
            fill = (f" fill: hole_frac_window {f['hole_frac_window']:.5f} changed {f['changed_px_window']}/{f['hole_px_window']}"
                    f" full_rows_unfilled {f['full_row_runs_left_unfilled']} lat|d| mean "
                    f"{sum(w['fill_latent_mean_abs_diff'] for w in sd['windows']) / len(sd['windows']):.5f}")
    elif sd.get("fill"):
        errs.append("warp render carries a fill log")
    ok = not errs
    nfail += (not ok)
    L.append(f"{'PASS' if ok else 'FAIL'} {base:28s} windows {len(got)} paired {same}/{len(ref)} sigma_start {s0} calls {calls} "
             f"pads {pads} resid/sigma std {sum(w['resid_over_sigma_std'] for w in sd['windows']) / len(sd['windows']):.4f}"
             f"{fill}{'  ERR ' + '; '.join(errs) if errs else ''}")
L.append(f"SUMMARY checked={nchk} failed={nfail}")
with open(OUT, "a") as fh:
    fh.write("\n".join(L) + "\n")
print("\n".join(L))
