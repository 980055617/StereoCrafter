"""input_side lane: tables from score_input_v1.py JSONs (no GPU).  Every number is printed with its source file.
usage: table_v1.py <scores_dir> <out_txt> <ref_input> <input>[,<input>...] [--models origin,deliv]
  rows = inputs (e.g. deployed,R1,RO,RF), deltas vs <ref_input> (normally R1), per model, 4 clips + mean."""
import json
import os
import sys

import numpy as np

SD, OUT, REF = sys.argv[1], sys.argv[2], sys.argv[3]
INPUTS = sys.argv[4].split(",")
MODELS = ["origin", "deliv"]
for a in sys.argv[5:]:
    if a.startswith("--models="):
        MODELS = a.split("=", 1)[1].split(",")
CLIPS = ["0301", "0204", "0052", "0147"]
assert not os.path.exists(OUT), f"refusing to overwrite {OUT}"
J = {}
for c in CLIPS:
    for inp in INPUTS:
        p = os.path.join(SD, f"{c}__{inp}.json")
        if os.path.exists(p):
            J[(c, inp)] = (json.load(open(p)), p)
L = []
P = lambda s="": L.append(s)
P(f"input_side tables from {SD}  (score_input_v1.py JSONs: <clip>__<input>.json)  ref input = {REF}")
P("LPIPS-Alex, SCORE_STEP=4; UNREG = score_clip_ll.py verbatim; REG_* = GT registered to each input's OWN warped window")
METRICS = [("UNREG", "lpips_clip", "UNREG"), ("REG_FRAME", "lpips_clip", "REG_FRAME"), ("REG_CLIP", "lpips_clip", "REG_CLIP"),
           ("BLK_LOCAL", "lpips_clip", "BLK_LOCAL")]
for m in MODELS:
    for name, key, sub in METRICS:
        P()
        P(f"--- {m}: {name} LPIPS (lower = better) ---")
        P(f"{'input':10s} " + " ".join(f"{c:>8s}" for c in CLIPS) + f" {'mean4':>8s} | " +
          " ".join(f"d{c:>7s}" for c in CLIPS) + f" {'dmean4':>8s} {'impr':>5s}")
        ref = {c: J[(c, REF)][0]["configs"][m][key][sub] for c in CLIPS if (c, REF) in J and m in J[(c, REF)][0]["configs"]}
        for inp in INPUTS:
            v = {c: J[(c, inp)][0]["configs"][m][key][sub] for c in CLIPS if (c, inp) in J and m in J[(c, inp)][0]["configs"]}
            if not v:
                continue
            row = f"{inp:10s} " + " ".join(f"{v[c]:8.4f}" if c in v else f"{'-':>8s}" for c in CLIPS)
            full = len(v) == 4
            row += f" {np.mean(list(v.values())):8.4f}" if full else f" {'(n<4)':>8s}"
            if inp != REF and len(ref) == 4 and full:
                d = {c: v[c] - ref[c] for c in CLIPS}
                row += " | " + " ".join(f"{d[c]:+8.4f}" for c in CLIPS) + f" {np.mean(list(d.values())):+8.4f} {sum(x < 0 for x in d.values()):>3d}/4"
            L.append(row)
    P()
    P(f"--- {m}: decomposition of the OUTPUT (GT regions from the REG_FRAME GT) and registration-free self metrics ---")
    P(f"{'input':10s} {'clip':5s} {'edgeHF/GT':>9s} {'stripeE/GT':>10s} {'flatHF/GT':>9s} {'halo%':>7s} {'sharp/GT':>8s} "
      f"{'selfEdge/GT':>11s} {'selfFlat/GT':>11s} {'rPSNR_REG':>9s}")
    for inp in INPUTS:
        for c in CLIPS:
            if (c, inp) not in J or m not in J[(c, inp)][0]["configs"]:
                continue
            cf = J[(c, inp)][0]["configs"][m]
            r, sr = cf["decomp_ratio"], cf["self_ratio"]
            P(f"{inp:10s} {c:5s} {r['edgeHF']:9.3f} {r['stripeE']:10.3f} {r['flatHF']:9.3f} {cf['decomp']['haloFrac']:7.2f} "
              f"{sr['sharp']:8.3f} {sr['selfEdgeHF']:11.3f} {sr['selfFlatHF']:11.3f} {cf['rPSNR']['REG_FRAME']:9.3f}")
P()
P("--- INPUT (the model's warped window) decomposition + registration of the real right eye to it ---")
P(f"{'input':10s} {'clip':5s} {'holes%':>7s} {'edgeHF/GT':>9s} {'stripeE/GT':>10s} {'flatHF/GT':>9s} {'sharp/GT':>8s} "
  f"{'regClip(dy,dx)':>15s} {'PSNRzero':>8s} {'PSNRclip':>8s} {'PSNRframe':>9s}")
for inp in INPUTS:
    for c in CLIPS:
        if (c, inp) not in J:
            continue
        j = J[(c, inp)][0]
        r, sr, rg = j["input_decomp_ratio"], j["input_self_ratio"], j["reg"]
        P(f"{inp:10s} {c:5s} {100*j['input_hole_frac']:7.3f} {r['edgeHF']:9.3f} {r['stripeE']:10.3f} {r['flatHF']:9.3f} "
          f"{sr['sharp']:8.3f} {('(%+d,%+d)' % (rg['clip_ddy'], rg['clip_ddx'])):>15s} {rg['zero_mean_psnr']:8.3f} "
          f"{rg['clip_mean_psnr']:8.3f} {rg['smooth_mean_psnr']:9.3f}")
P()
P("sources:")
for (c, inp), (_, p) in sorted(J.items()):
    P(f"  {c} {inp}: {p}")
open(OUT, "w").write("\n".join(L) + "\n")
print("\n".join(L))
