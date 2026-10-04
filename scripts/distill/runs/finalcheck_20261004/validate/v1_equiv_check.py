"""V1 equivalence gate (PREREG.txt): the copy's 0204 rows must equal the TRACKED score_temporal.py's 0204 rows at
the PRINTED precision of the scripts (tLP/warp/seam %7.4f, nonseam %8.4f -> 4 decimals; leftPSNR is not printed by
the tracked script, compared at 2 decimals as score_clip prints it).  Exact float equality is reported separately
(informational) -- the gate is the pre-registered printed-digit rule."""
import json, sys
O = "outputs/finalcheck_20261004/validate/v1_temporal"
a = json.load(open(f"{O}/control_tracked_0204.json"))["0204"]
b = json.load(open(f"{O}/temporal_576_12clip.json"))["0204"]
bad = 0; inexact = 0
for tag in ("GT", "0204_origin_ll", "0204_mstudent2_step800_deliv_ll"):
    for k in ("tLP", "warp", "seam", "nonseam") + (("leftPSNR",) if tag != "GT" else ()):
        x, y = a[tag][k], b[tag][k]
        nd = 2 if k == "leftPSNR" else 4
        printed_same = f"{x:.{nd}f}" == f"{y:.{nd}f}"
        exact = (x == y)
        bad += (not printed_same); inexact += (not exact)
        print(f"{tag:34s} {k:9s} tracked={x!r:24s} copy={y!r:24s} printed({nd}dp) {'SAME' if printed_same else 'DIFF'}"
              f" | exact {'IDENTICAL' if exact else 'DIFF %.3g' % (y - x)}")
print(f"exact-float identity: {'ALL IDENTICAL' if inexact == 0 else f'{inexact} values differ in the last bits (informational)'}")
print("EQUIV_PASS" if bad == 0 else f"EQUIV_FAIL ({bad} values differ at printed precision)")
sys.exit(0 if bad == 0 else 4)
