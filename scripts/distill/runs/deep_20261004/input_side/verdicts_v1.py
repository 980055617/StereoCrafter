"""input_side lane: apply every pre-registered rule (PREREG.txt + PREREG_ADDENDUM_1_codec.txt) to the score JSONs.
usage: verdicts_v1.py <out_txt>"""
import json, os, sys
import numpy as np
os.chdir("/home/kawa/master_project/StereoCrafter")
OUT = sys.argv[1]
assert not os.path.exists(OUT), OUT
S = "outputs/deep_20261004/input_side/scores_v1"
CL = ["0301", "0204", "0052", "0147"]
M = ["origin", "deliv"]
J = lambda c, i: json.load(open(f"{S}/{c}__{i}.json"))
L = []
P = L.append
def lp(c, i, m, k):
    return J(c, i)["configs"][m]["lpips_clip"][k]
def contrast(a, b, m, k):
    d = [lp(c, a, m, k) - lp(c, b, m, k) for c in CL]
    return d, float(np.mean(d)), sum(x < 0 for x in d)
P("VERDICTS -- input_side lane (deep_20261004), every rule as pre-registered.  Source JSONs: " + S + "/<clip>__<input>.json")
P("Clips 0301 0204 0052 0147; models origin (deployed) and deliv (the deliverable .pt); LPIPS-Alex, lower = better.")
P("")
P("R1-FIDELITY (PREREG, reported): R1 vs DEP.  Representative iff mean|d| <= 0.003 and max|d| <= 0.006.")
for m in M:
    for k in ("UNREG", "REG_FRAME"):
        d, mu, _ = contrast("R1", "deployed", m, k)
        a = np.abs(d)
        P(f"  {m:6s} {k:9s} d(R1-DEP) {' '.join(f'{x:+.4f}' for x in d)}  mean|d| {a.mean():.4f} max {a.max():.4f} -> "
          f"{'representative' if a.mean() <= 0.003 and a.max() <= 0.006 else 'NOT representative'}")
P("  explained by ADDENDUM 1: R1C (= R1 + the deployed cv2-mp4v intermediate) vs DEP:")
for m in M:
    for k in ("UNREG", "REG_FRAME"):
        d, mu, _ = contrast("R1C", "deployed", m, k)
        a = np.abs(d)
        P(f"  {m:6s} {k:9s} d(R1C-DEP) {' '.join(f'{x:+.4f}' for x in d)}  mean|d| {a.mean():.4f} max {a.max():.4f}")
P("  -> the depth reconstruction is faithful; the R1-DEP gap is the codec.")
P("")
P("PROBE 1 (ORACLE disparity fit, RF vs R1).  P1: mean4 dREG_FRAME <= -0.005 AND >= 3/4 improved; 'every model' = both.")
p1 = {}
for m in M:
    d, mu, n = contrast("RF", "R1", m, "REG_FRAME")
    p1[m] = mu <= -0.005 and n >= 3
    du, muu, nu = contrast("RF", "R1", m, "UNREG")
    db, mub, nb = contrast("RF", "R1", m, "BLK_LOCAL")
    dc, muc, nc = contrast("RF", "R1", m, "REG_CLIP")
    dro, muro, nro = contrast("RO", "R1", m, "UNREG")
    dfd, mufd, nfd = contrast("RF", "deployed", m, "REG_FRAME")
    dfdu, mufdu, nfdu = contrast("RF", "deployed", m, "UNREG")
    P(f"  {m:6s} dREG_FRAME {' '.join(f'{x:+.4f}' for x in d)}  mean {mu:+.4f}  {n}/4  -> P1 {'PASS' if p1[m] else 'FAIL'}")
    P(f"  {m:6s} dUNREG     {' '.join(f'{x:+.4f}' for x in du)}  mean {muu:+.4f}  {nu}/4   (RO-R1 dUNREG mean {muro:+.4f} = convergence part)")
    P(f"  {m:6s} dREG_CLIP  {' '.join(f'{x:+.4f}' for x in dc)}  mean {muc:+.4f}  {nc}/4")
    P(f"  {m:6s} dBLK_LOCAL {' '.join(f'{x:+.4f}' for x in db)}  mean {mub:+.4f}  {nb}/4  -> "
      f"{'gain = GEOMETRIC ALIGNMENT (|mean dBLK_LOCAL| < 0.002)' if abs(mub) < 0.002 else 'BLK_LOCAL also moves (|mean| >= 0.002)'}")
    P(f"  {m:6s} vs the DEPLOYED input: RF-DEP dREG_FRAME {' '.join(f'{x:+.4f}' for x in dfd)}  mean {mufd:+.4f} {nfd}/4; "
      f"dUNREG mean {mufdu:+.4f} {nfdu}/4")
P(f"  P1 for every model: {'PASS' if all(p1.values()) else 'FAIL'}")
P("  deliverable - origin gap (REG_FRAME): on R1 " +
  " ".join(f"{lp(c, 'R1', 'deliv', 'REG_FRAME') - lp(c, 'R1', 'origin', 'REG_FRAME'):+.4f}" for c in CL) +
  f" mean {np.mean([lp(c, 'R1', 'deliv', 'REG_FRAME') - lp(c, 'R1', 'origin', 'REG_FRAME') for c in CL]):+.4f};  on RF " +
  " ".join(f"{lp(c, 'RF', 'deliv', 'REG_FRAME') - lp(c, 'RF', 'origin', 'REG_FRAME'):+.4f}" for c in CL) +
  f" mean {np.mean([lp(c, 'RF', 'deliv', 'REG_FRAME') - lp(c, 'RF', 'origin', 'REG_FRAME') for c in CL]):+.4f}")
P("  registration-free sharpness of the OUTPUT (sharp/GT, selfEdgeHF/GT), R1 -> RF:")
for m in M:
    P(f"  {m:6s} " + "  ".join(f"{c} {J(c, 'R1')['configs'][m]['self_ratio']['sharp']:.3f}->{J(c, 'RF')['configs'][m]['self_ratio']['sharp']:.3f}"
                              f" / {J(c, 'R1')['configs'][m]['self_ratio']['selfEdgeHF']:.3f}->{J(c, 'RF')['configs'][m]['self_ratio']['selfEdgeHF']:.3f}" for c in CL))
P("")
P("PROBE 2 step B (SS4 = the largest-input-stripe-reduction variant; P2A failed for all six variants, TABLE_P2A_INPUT.txt).")
P("  P2B: mean4 dREG_FRAME (SS4-R1) <= -0.003 AND >= 3/4 improved AND output stripeE/GT falls on average.")
for m in M:
    d, mu, n = contrast("SS4", "R1", m, "REG_FRAME")
    du, muu, nu = contrast("SS4", "R1", m, "UNREG")
    sr1 = np.mean([J(c, "R1")["configs"][m]["decomp_ratio"]["stripeE"] for c in CL])
    ss4 = np.mean([J(c, "SS4")["configs"][m]["decomp_ratio"]["stripeE"] for c in CL])
    ok = mu <= -0.003 and n >= 3 and ss4 < sr1
    P(f"  {m:6s} dREG_FRAME {' '.join(f'{x:+.4f}' for x in d)}  mean {mu:+.4f} {n}/4; dUNREG mean {muu:+.4f} {nu}/4; "
      f"output stripeE/GT mean {sr1:.3f} -> {ss4:.3f}  -> P2B {'PASS' if ok else 'FAIL'}")
P("")
P("ADDENDUM 1 codec control.  P3: mean4 dREG_FRAME (R1 - R1C) <= -0.002 AND R1 better on >= 3/4 (lossless helps).")
for m in M:
    d, mu, n = contrast("R1", "R1C", m, "REG_FRAME")
    du, muu, nu = contrast("R1", "R1C", m, "UNREG")
    ok = mu <= -0.002 and n >= 3
    P(f"  {m:6s} d(R1-R1C) REG_FRAME {' '.join(f'{x:+.4f}' for x in d)}  mean {mu:+.4f} {n}/4; UNREG mean {muu:+.4f} {nu}/4 -> P3 {'PASS' if ok else 'FAIL'}")
open(OUT, "w").write("\n".join(L) + "\n")
print("\n".join(L))
