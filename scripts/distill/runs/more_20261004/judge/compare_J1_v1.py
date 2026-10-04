#!/usr/bin/env python
"""judge J1c/J1d comparison (CPU): my re-scores vs the temporal lane's numbers.  usage: compare_J1_v1.py OUT.txt"""
import glob, json, os, sys
os.chdir("/home/kawa/master_project/StereoCrafter")
J = "scripts/distill/runs/more_20261004/judge"; T = "scripts/distill/runs/more_20261004/temporal"
TS = "outputs/more_20261004/temporal/scores"; MY = "outputs/more_20261004/judge/rescore_temporal"
CL = ["0170", "0259", "0042", "0301"]
ROWS = ["origin_ll", "deliv_g100_T5nat", "T5nat_dcs14", "origin_g101_s8_dcs14"]
def rows(path):
    out = {}
    for ln in open(path):
        if ln.startswith("ROW "):
            d = dict(kv.split("=", 1) for kv in ln.split()[1:]); out[(d["clip"], d["tag"][len(d["clip"]) + 1:])] = d
    return out
lane = {}
for f in glob.glob(f"{T}/SCORES_LPIPS_*.txt"):
    for k, v in rows(f).items():
        if k in lane: assert abs(float(lane[k]["lpips"]) - float(v["lpips"])) < 1e-9, (k, f)
        lane[k] = v
mine = {}
for c in CL: mine.update(rows(f"{J}/SCORES_J1c_LPIPS_{c}.txt"))
lw = {}
for f in glob.glob(f"{TS}/*.json"):
    for c, d in json.load(open(f)).items():
        if not isinstance(d, dict): continue
        for tag, v in d.items():
            if not isinstance(v, dict): continue
            if "warp" in v: lw.setdefault((c, tag[len(c) + 1:] if tag != "GT" else "GT"), set()).add(round(v["warp"], 12))
mw = {}
for c in CL:
    for tag, v in json.load(open(f"{MY}/J1d_warp_{c}.json"))[c].items():
        mw[(c, tag[len(c) + 1:] if tag != "GT" else "GT")] = v
L = ["judge J1c/J1d -- re-scores of the temporal lane's dcs14 key rows vs the lane's own numbers",
     "LPIPS: score_clip_ll.py SCORE_STEP=4 (mine: SCORES_J1c_LPIPS_<clip>.txt; lane: temporal/SCORES_LPIPS_*.txt)",
     "warp : mine = validate-lane score_temporal_ll.py (independent copy), lane = temporal/score_temporal_tr_v1.py JSONs", ""]
nf = 0; mean = {}
for r in ROWS:
    for c in CL:
        a = float(mine[(c, r)]["lpips"]); b = float(lane[(c, r)]["lpips"]); dl = a - b
        wa = mw[(c, r)]["warp"]; wbs = lw[(c, r)]; wb = sorted(wbs)[0]; dw = wa - wb
        ok = abs(dl) <= 1e-4 and abs(dw) <= 2e-5 and len(wbs) == 1
        nf += not ok
        mean.setdefault(r, []).append((a, wa))
        L.append(f"{'PASS' if ok else 'FAIL'} {c} {r:24s} LPIPS mine {a:.6f} lane {b:.6f} (d {dl:+.1e})   warp mine {wa:.6f} lane {wb:.6f} (d {dw:+.1e}; lane values {len(wbs)})")
L.append("")
o = [x[1] for x in mean["origin_ll"]]; om = sum(o) / 4
for r in ROWS:
    lp = sum(x[0] for x in mean[r]) / 4; wm = sum(x[1] for x in mean[r]) / 4
    L.append(f"mean4 {r:24s} LPIPS {lp:.4f}   warp {wm:.5f}   gap vs origin_ll {100*(wm/om-1):+.2f} %")
b = sum(x[1] for x in mean["deliv_g100_T5nat"]) / 4; d14 = sum(x[1] for x in mean["T5nat_dcs14"]) / 4; o14 = sum(x[1] for x in mean["origin_g101_s8_dcs14"]) / 4
L.append(f"dcs14 vs origin at the same decode: {100*(d14/o14-1):+.2f} %  (BASE vs deployed origin {100*(b/om-1):+.2f} %)")
for r, base in [("T5nat_dcs14", "deliv_g100_T5nat"), ("origin_g101_s8_dcs14", "origin_ll")]:
    per = [f"{c}:{mean[r][i][0]-mean[base][i][0]:+.4f}/{mean[r][i][1]/mean[base][i][1]:.3f}" for i, c in enumerate(CL)]
    L.append(f"{r} vs {base} per clip (dLPIPS / warp ratio): " + " ".join(per))
L.append(f"SUMMARY rows={len(ROWS)*len(CL)} failed={nf}")
open(sys.argv[1], "w").write("\n".join(L) + "\n"); print("\n".join(L))
