#!/usr/bin/env python
"""judge J1b: per-window initial-latent fingerprints (speed_log.json 'windows'[k]['init_md5']) of the temporal lane's
dcs14 renders vs their paired references.  CPU only, read-only.  usage: check_pairing_v1.py OUT.txt"""
import json, sys, os
os.chdir("/home/kawa/master_project/StereoCrafter")
T = "outputs/more_20261004/temporal/clips"
F = "outputs/finalcheck_20261004/speed/clips"
def fp(d):
    j = json.load(open(f"{d}/speed_log.json"))
    return [w["init_md5"] for w in j["windows"]], j
rows, nfail = [], 0
for c in ["0170", "0259", "0042", "0301"]:
    pairs = [(f"{T}/{c}_T5nat_dcs14", f"{F}/{c}_deliv_g100_T5nat", "dcs14 vs BASE")]
    ref_o = f"{F}/{c}_origin_g101_s8" if os.path.exists(f"{F}/{c}_origin_g101_s8/speed_log.json") else f"{F}/{c}_origin_g100_s8"
    pairs.append((f"{T}/{c}_origin_g101_s8_dcs14", ref_o, "origin_dcs14 vs 8-step origin"))
    for a, b, what in pairs:
        fa, ja = fp(a); fb, jb = fp(b)
        same = sum(x == y for x, y in zip(fa, fb))
        ok = same == len(fa) == len(fb)
        nfail += (not ok)
        rows.append(f"{'PASS' if ok else 'FAIL'} {c} {what}: {same}/{len(fa)} windows identical (n {len(fa)} vs {len(fb)}) "
                    f"| {a} [dcs={ja.get('dcs')} guid={ja['guid']} steps={ja['steps']}] vs {b} [guid={jb['guid']} steps={jb['steps']} sigmas={'T5' if jb.get('sigmas') else 'default'}]")
rows.append(f"SUMMARY pairs={len(rows)} failed={nfail}")
open(sys.argv[1], "w").write("\n".join(rows) + "\n"); print("\n".join(rows))
