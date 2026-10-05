#!/usr/bin/env python
"""deep_20261004 / sdedit lane: stage table + pre-registered verdicts (PREREG.txt).  CPU only, reads scorer outputs.
usage: table_v1.py <stage_dir> <rule P4|P12|SMOKE> OUT.txt OUT.json
Every number printed comes from the stage dir's files (paths listed in the PROVENANCE block):
  SCORES.txt (score_clip_ll.py ROW lines: UNREG, sharp) | reg/<clip>.json, reg_wide/0125.json (score_registered_v1[w])
  decomp/decomp.json (stripeE, edgeHF, ...) | temporal/temporal.json (RAFT warp error, tLP, seam ratio)
  clips/<clip>_<label>/speed_log.json (UNet evaluations, UNet GPU seconds)
"""
import json
import math
import os
import re
import sys
from collections import OrderedDict

os.chdir("/home/kawa/master_project/StereoCrafter")
SD, RULE, OUTT, OUTJ = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
for p in (OUTT, OUTJ):
    if os.path.exists(p):
        sys.exit(f"refusing to overwrite {p}")
MODELS = OrderedDict(origin=("origin_ll", "origin_g100_s8"), deliv=("mstudent2_step800_deliv_ll", "deliv_g100_s8"))
VARS = ["sd31", "sd7", "sd1", "sd1fill"]
DESCR = ["INPUT_warp", "INPUT_fill"]
PUB_UNREG_FILE = "scripts/distill/runs/finalcheck_20261004/speed/SCORES_STEP1.txt"
PUB_TEMP_FILE = "scripts/distill/runs/finalcheck_20261004/validate/V1_TEMPORAL_576_12CLIP.txt"
out = []


def P(*a):
    s = " ".join(str(x) for x in a)
    out.append(s)
    print(s, flush=True)


# ------------------------------------------------------------------ load
rows = json.load(open(f"{SD}/ROWS.json"))
clips = rows["clips"]
unreg, sharp, gtsharp = {}, {}, {}
for line in open(f"{SD}/SCORES.txt"):
    if line.startswith("ROW "):
        kv = dict(t.split("=", 1) for t in line.split()[1:])
        lab = kv["tag"][len(kv["clip"]) + 1:]
        unreg[(kv["clip"], lab)] = float(kv["lpips"])
        sharp[(kv["clip"], lab)] = float(kv["sharp"])
        gtsharp[kv["clip"]] = float(kv["gtSharp"])
reg, regsrc = {}, {}
for c in clips:
    p = f"{SD}/reg/{c}.json"
    d = json.load(open(p))
    for lab, v in d["configs"].items():
        reg[(c, lab)] = dict(v["lpips_clip"])
        regsrc[(c, lab)] = p
    if c == "0125" and os.path.exists(f"{SD}/reg_wide/0125.json"):
        dw = json.load(open(f"{SD}/reg_wide/0125.json"))
        for lab, v in dw["configs"].items():
            reg[(c, lab)]["REG_FRAME_NARROW"] = reg[(c, lab)]["REG_FRAME"]
            reg[(c, lab)]["REG_FRAME"] = v["lpips_clip"]["REG_FRAME"]          # widened grid = primary (addendum)
            reg[(c, lab)]["REG_CLIP_WIDE"] = v["lpips_clip"]["REG_CLIP"]
            regsrc[(c, lab)] = f"{SD}/reg_wide/0125.json (REG_FRAME) + {p}"
dec = json.load(open(f"{SD}/decomp/decomp.json")) if os.path.exists(f"{SD}/decomp/decomp.json") else {}
tem = json.load(open(f"{SD}/temporal/temporal.json")) if os.path.exists(f"{SD}/temporal/temporal.json") else {}


def dget(c, lab, k):
    try:
        return float(dec[c]["rows"][lab][k])
    except KeyError:
        return float("nan")


def tget(c, lab, k):
    try:
        return float(tem[c][f"{c}_{lab}"][k])
    except KeyError:
        return float("nan")


def cost(c, lab):
    for root in ("outputs/deep_20261004/sdedit/clips", "outputs/finalcheck_20261004/speed/clips"):
        p = f"{root}/{c}_{lab}/speed_log.json"
        if os.path.exists(p):
            s = json.load(open(p))
            return s["unet_calls_total"], s["unet_ms_sum"] / 1000.0, s["n_windows"]
    return None


def fmt(x, w=8, d=4):
    return f"{x:{w}.{d}f}" if (x is not None and x == x) else " " * (w - 3) + "n/a"


def mean(a):
    a = [x for x in a if x == x]
    return sum(a) / len(a) if a else float("nan")


present = lambda lab: all((c, lab) in reg for c in clips)
labs_all = [lab for lab in rows["labels"]]
P("=" * 118)
P(f"deep_20261004 / sdedit -- stage {os.path.basename(SD)}  rule {RULE}  clips {' '.join(clips)}")
P("SDEdit start x = 0.18215*z_src + sigma_start*eps (eps = the deployed window's own initial noise), guidance 1.00, lossless FFV1")
P("=" * 118)

# ------------------------------------------------------------------ K5 / K6 reproduction gates
pub = {}
for line in open(PUB_UNREG_FILE):
    if line.startswith("ROW "):
        kv = dict(t.split("=", 1) for t in line.split()[1:])
        pub[(kv["clip"], kv["tag"][len(kv["clip"]) + 1:])] = float(kv["lpips"])
k5 = []
for c in clips:
    for lab in ("origin_ll", "mstudent2_step800_deliv_ll", "origin_g100_s8", "deliv_g100_s8"):
        if (c, lab) in unreg and (c, lab) in pub:
            k5.append((c, lab, "UNREG", unreg[(c, lab)], pub[(c, lab)]))
    er = f"outputs/more_20261004/eval_robustness/score_v1/{c}.json"
    if os.path.exists(er):
        e = json.load(open(er))
        for lab in ("origin_ll", "mstudent2_step800_deliv_ll"):
            for var in ("REG_CLIP",) + (("REG_FRAME",) if c != "0125" else ("REG_FRAME_NARROW",)):
                ev = e["configs"][lab]["lpips_clip"]["REG_FRAME" if var == "REG_FRAME_NARROW" else var]
                k5.append((c, lab, var, reg[(c, lab)].get(var, float("nan")), ev))
    if c == "0125":
        ew = json.load(open("outputs/more_20261004/eval_robustness/score_v1_wide/0125.json"))
        for lab in ("origin_ll", "mstudent2_step800_deliv_ll"):
            k5.append((c, lab, "REG_FRAME(wide)", reg[(c, lab)]["REG_FRAME"], ew["configs"][lab]["lpips_clip"]["REG_FRAME"]))
k5_fail = [r for r in k5 if not abs(r[3] - r[4]) <= 1e-4]
P(f"K5 scorer reproduction: {len(k5) - len(k5_fail)}/{len(k5)} cells within +-0.0001 of the published values "
  f"({PUB_UNREG_FILE}; outputs/more_20261004/eval_robustness/score_v1[_wide]/<clip>.json)  -> {'PASS' if not k5_fail else 'FAIL'}")
for r in k5_fail:
    P(f"   K5 FAIL {r}")
ptxt = open(PUB_TEMP_FILE).read()
m = re.search(r"--- warp error ---\n.*?\n(?:.*\n)*?  origin\s+([\d. ]+)\n", ptxt)
pw = dict(zip("0042 0052 0125 0128 0141 0147 0170 0204 0225 0251 0259 0301".split(), [float(x) for x in m.group(1).split()[:12]]))
k6 = [(c, tget(c, "origin_ll", "warp"), pw[c]) for c in clips if tem]
k6_fail = [r for r in k6 if not (r[1] == r[1] and round(r[1], 4) == r[2])]
P(f"K6 temporal reproduction (origin_ll warp error vs {PUB_TEMP_FILE}, 4 dp): {len(k6) - len(k6_fail)}/{len(k6)} -> "
  f"{'PASS' if (k6 and not k6_fail) else ('FAIL' if k6 else 'not run')}  {[(c, round(a, 4), b) for c, a, b in k6]}")
dk5 = re.findall(r"K5 decomposition reproduction.*?: (PASS|FAIL)", open(f"{SD}/decomp/DECOMP.txt").read()) \
    if os.path.exists(f"{SD}/decomp/DECOMP.txt") else []
P(f"K5 decomposition (run_decomp's own check vs outputs/review_20261001/METRICS_WHOLEFRAME.txt, clips the review covered): {dk5 or 'none covered'}")

# ------------------------------------------------------------------ per-clip absolute table
def block(title, key, labs, w=8, d=4, src=None):
    P("")
    P(f"--- {title} ---")
    P(f"  {'row':30s} " + " ".join(f"{c:>{w}s}" for c in clips) + f" {'MEAN':>{w}s}")
    for lab in labs:
        vals = [src(c, lab) for c in clips]
        P(f"  {lab:30s} " + " ".join(fmt(v, w, d) for v in vals) + " " + fmt(mean(vals), w, d))


order = []
for mname, (dep, ctl) in MODELS.items():
    order += [dep, ctl] + [f"{mname}_g100_{v}" for v in VARS]
order = [lab for lab in order if any((c, lab) in unreg for c in clips)]
descr = [lab for lab in DESCR if any((c, lab) in unreg for c in clips)]
block("REGISTERED LPIPS, REG_FRAME (PRIMARY; per-frame GT shift; 0125 widened grid) -- lower is better", "REG_FRAME",
      order + descr, src=lambda c, l: reg.get((c, l), {}).get("REG_FRAME", float("nan")))
block("REGISTERED LPIPS, REG_CLIP (one GT shift per clip; robustness)", "REG_CLIP", order + descr,
      src=lambda c, l: reg.get((c, l), {}).get("REG_CLIP", float("nan")))
block("UNREGISTERED LPIPS (published scorer, score_clip_ll.py)", "UNREG", order + descr,
      src=lambda c, l: unreg.get((c, l), float("nan")))
block("FLAT-REGION STRIPE ENERGY stripeE (mean |horizontal diff| in GT-flat pixels; GT row = the real right eye)", "stripeE",
      ["GT"] + order + descr + ["INPUT warped (none)"], w=8, d=5, src=lambda c, l: dget(c, l, "stripeE"))
block("EDGE ENERGY AT GT EDGES edgeHF (mean |Laplacian| in GT top-decile gradient pixels)", "edgeHF",
      ["GT"] + order + descr + ["INPUT warped (none)"], w=8, d=5, src=lambda c, l: dget(c, l, "edgeHF"))
block("stripeE near holes (GT-flat pixels within 2 px of a hole)", "near", order + descr, w=8, d=5,
      src=lambda c, l: dget(c, l, "stripeE_nearHole"))
block("RAFT WARP ERROR (lower = more temporally stable; GT row = the real right eye)", "warp", order + descr, w=8, d=4,
      src=lambda c, l: tget(c, l, "warp"))
block("sharpness / GT sharpness (score_clip_ll.py 'sharp')", "sharp", order + descr, w=8, d=3,
      src=lambda c, l: sharp.get((c, l), float("nan")) / gtsharp[c] if (c, l) in sharp else float("nan"))

# ------------------------------------------------------------------ contrasts and verdicts
res = OrderedDict()
P("")
P("=" * 118)
P(f"CONTRASTS vs THE SAME MODEL'S DEPLOYED RENDER (negative LPIPS delta = better; ratios: 1.00 = unchanged)")
P("=" * 118)
for mname, (dep, ctl) in MODELS.items():
    for v in VARS:
        lab = f"{mname}_g100_{v}"
        if not present(lab):
            continue
        r = OrderedDict(label=lab, model=mname, deployed=dep, control=ctl, clips=clips)
        r["dREG_FRAME"] = [reg[(c, lab)]["REG_FRAME"] - reg[(c, dep)]["REG_FRAME"] for c in clips]
        r["dREG_CLIP"] = [reg[(c, lab)]["REG_CLIP"] - reg[(c, dep)]["REG_CLIP"] for c in clips]
        r["dUNREG"] = [unreg[(c, lab)] - unreg[(c, dep)] for c in clips]
        r["dREG_FRAME_vs_ctl"] = [reg[(c, lab)]["REG_FRAME"] - reg[(c, ctl)]["REG_FRAME"] for c in clips]
        # PREREG_ADDENDUM_1 A1: contrast with ORIGIN (deployed origin_ll) for every row -- reported, not a P4/P12 gate
        r["dREG_FRAME_vs_origin"] = [reg[(c, lab)]["REG_FRAME"] - reg[(c, "origin_ll")]["REG_FRAME"] for c in clips]
        r["dREG_CLIP_vs_origin"] = [reg[(c, lab)]["REG_CLIP"] - reg[(c, "origin_ll")]["REG_CLIP"] for c in clips]
        r["dUNREG_vs_origin"] = [unreg[(c, lab)] - unreg[(c, "origin_ll")] for c in clips]
        r["n_improved_vs_origin"] = sum(x < 0 for x in r["dREG_FRAME_vs_origin"])
        r["round_rule_vs_origin"] = dict(
            mean_dREG_FRAME=mean(r["dREG_FRAME_vs_origin"]), n_improved=r["n_improved_vs_origin"], n=len(clips),
            met=(len(clips) == 12 and mean(r["dREG_FRAME_vs_origin"]) <= -0.005 and r["n_improved_vs_origin"] >= 9),
            note="judged only at 12 clips (A1); descriptive at fewer")
        r["dUNREG_vs_ctl"] = [unreg[(c, lab)] - unreg[(c, ctl)] for c in clips]
        r["stripe_ratio"] = [dget(c, lab, "stripeE") / dget(c, dep, "stripeE") for c in clips]
        r["edge_ratio"] = [dget(c, lab, "edgeHF") / dget(c, dep, "edgeHF") for c in clips]
        r["near_ratio"] = [dget(c, lab, "stripeE_nearHole") / dget(c, dep, "stripeE_nearHole") for c in clips]
        r["warp_ratio"] = [tget(c, lab, "warp") / tget(c, dep, "warp") for c in clips]
        r["n_improved_REG_FRAME"] = sum(x < 0 for x in r["dREG_FRAME"])
        r["n_improved_REG_CLIP"] = sum(x < 0 for x in r["dREG_CLIP"])
        r["n_improved_UNREG"] = sum(x < 0 for x in r["dUNREG"])
        r["n_improved_vs_ctl"] = sum(x < 0 for x in r["dREG_FRAME_vs_ctl"])
        r["mean_stripe_ratio"] = mean(r["stripe_ratio"])
        r["clips_stripe_gt_1.10"] = [c for c, x in zip(clips, r["stripe_ratio"]) if x > 1.10]
        r["mean_edge_ratio"] = mean(r["edge_ratio"])
        r["mean_warp_ratio"] = mean(r["warp_ratio"])
        r["unreg_disagrees"] = [c for c, a, b in zip(clips, r["dREG_FRAME"], r["dUNREG"]) if (a < 0) != (b < 0)]
        n = len(clips)
        need = {"P4": 3, "P12": 9}.get(RULE)
        if need is not None:
            lp = r["n_improved_REG_FRAME"] >= need
            st = r["mean_stripe_ratio"] <= 1.10
            r["verdict"] = dict(rule=RULE, lpips_part=lp, stripe_part=st, PASS=bool(lp and st),
                                need=f">= {need}/{n} REG_FRAME improved and mean stripeE ratio <= 1.10")
        cst = [cost(c, lab) for c in clips]
        r["unet_calls_per_window"] = sorted({(x[0] // x[2]) for x in cst if x})
        res[lab] = r
        P("")
        P(f"[{lab}]  vs {dep} (deployed)  |  SDEdit effect vs {ctl} (guidance-1.00 control)  |  UNet evals/window "
          f"{r['unet_calls_per_window']} (deployed 8)")
        P(f"  {'':24s} " + " ".join(f"{c:>8s}" for c in clips) + f" {'MEAN':>8s}  improved")
        for k, nm, d in (("dREG_FRAME", "dLPIPS REG_FRAME", 4), ("dREG_CLIP", "dLPIPS REG_CLIP", 4),
                         ("dUNREG", "dLPIPS UNREG", 4), ("dREG_FRAME_vs_ctl", "dREG_FRAME vs ctl", 4),
                         ("dUNREG_vs_ctl", "dUNREG vs ctl", 4), ("dREG_FRAME_vs_origin", "dREG_FRAME vs ORIGIN", 4),
                         ("dREG_CLIP_vs_origin", "dREG_CLIP vs ORIGIN", 4), ("dUNREG_vs_origin", "dUNREG vs ORIGIN", 4)):
            vals = r[k]
            P(f"  {nm:24s} " + " ".join(f"{x:+8.{d}f}" for x in vals) + f" {mean(vals):+8.{d}f}  "
              f"{sum(x < 0 for x in vals)}/{n}")
        for k, nm in (("stripe_ratio", "stripeE ratio"), ("near_ratio", "stripeE-near ratio"),
                      ("edge_ratio", "edgeHF ratio"), ("warp_ratio", "warp-error ratio")):
            vals = r[k]
            P(f"  {nm:24s} " + " ".join(fmt(x, 8, 3) for x in vals) + f" {fmt(mean(vals), 8, 3)}")
        if "verdict" in r:
            v_ = r["verdict"]
            P(f"  VERDICT {RULE}: {'PASS' if v_['PASS'] else 'FAIL'}  (REG_FRAME improved {r['n_improved_REG_FRAME']}/{n}: "
              f"{'ok' if v_['lpips_part'] else 'NO'}; mean stripeE ratio {r['mean_stripe_ratio']:.3f}: "
              f"{'ok' if v_['stripe_part'] else 'NO (> 1.10)'})")
        rr = r["round_rule_vs_origin"]
        P(f"  ROUND RULE vs ORIGIN (A1; judged at 12 clips only): mean dREG_FRAME {rr['mean_dREG_FRAME']:+.4f} "
          f"(need <= -0.0050), improved {rr['n_improved']}/{rr['n']} (need >= 9/12) -> "
          f"{('MET' if rr['met'] else 'not met') if len(clips) == 12 else 'descriptive at ' + str(len(clips)) + ' clips'}")
        P(f"  FLAGS: clips with stripeE ratio > 1.10: {r['clips_stripe_gt_1.10'] or 'none'}; mean edgeHF ratio "
          f"{r['mean_edge_ratio']:.3f}; mean warp-error ratio {r['mean_warp_ratio']:.3f}"
          f"{' (> 1.05 FLAG)' if r['mean_warp_ratio'] > 1.05 else ''}; UNREG direction differs on: "
          f"{r['unreg_disagrees'] or 'none'}; REG_CLIP improved {r['n_improved_REG_CLIP']}/{n}")

P("")
P("=" * 118)
P("SUMMARY (mean over clips; d = variant - same model's deployed render)")
P(f"  {'row':26s} {'REG_FRAME':>9s} {'dREG_FR':>8s} {'impr':>5s} {'dREG_CL':>8s} {'dUNREG':>8s} {'impr':>5s} "
  f"{'d vs ctl':>8s} {'d vs ORIG':>9s} {'impr':>5s} {'stripeR':>7s} {'edgeR':>6s} {'warpR':>6s} {'verdict':>8s}")
for lab, r in res.items():
    P(f"  {lab:26s} {mean([reg[(c, lab)]['REG_FRAME'] for c in clips]):9.4f} {mean(r['dREG_FRAME']):+8.4f} "
      f"{r['n_improved_REG_FRAME']:>2d}/{len(clips):<2d} {mean(r['dREG_CLIP']):+8.4f} {mean(r['dUNREG']):+8.4f} "
      f"{r['n_improved_UNREG']:>2d}/{len(clips):<2d} {mean(r['dREG_FRAME_vs_ctl']):+8.4f} "
      f"{mean(r['dREG_FRAME_vs_origin']):+9.4f} {r['n_improved_vs_origin']:>2d}/{len(clips):<2d} {r['mean_stripe_ratio']:7.3f} "
      f"{r['mean_edge_ratio']:6.3f} {r['mean_warp_ratio']:6.3f} "
      f"{(('PASS' if r['verdict']['PASS'] else 'FAIL') if 'verdict' in r else '-'):>8s}")
for mname, (dep, ctl) in MODELS.items():
    P(f"  {dep:26s} {mean([reg[(c, dep)]['REG_FRAME'] for c in clips]):9.4f}   (deployed reference)   UNREG "
      f"{mean([unreg[(c, dep)] for c in clips]):.4f}")
for lab in descr:
    if present(lab):
        dO = [reg[(c, lab)]["REG_FRAME"] - reg[(c, "origin_ll")]["REG_FRAME"] for c in clips]
        dD = [reg[(c, lab)]["REG_FRAME"] - reg[(c, "mstudent2_step800_deliv_ll")]["REG_FRAME"] for c in clips]
        sO = [dget(c, lab, "stripeE") / dget(c, "origin_ll", "stripeE") for c in clips]
        sD = [dget(c, lab, "stripeE") / dget(c, "mstudent2_step800_deliv_ll", "stripeE") for c in clips]
        P(f"  {lab:26s} {mean([reg[(c, lab)]['REG_FRAME'] for c in clips]):9.4f}   (descriptive: raw warped input) "
          f"d vs origin_ll {mean(dO):+.4f} ({sum(x < 0 for x in dO)}/{len(clips)} better), d vs deliv {mean(dD):+.4f} "
          f"({sum(x < 0 for x in dD)}/{len(clips)}); UNREG {mean([unreg[(c, lab)] for c in clips]):.4f}; stripeE ratio vs "
          f"origin_ll {mean(sO):.3f}, vs deliv {mean(sD):.3f}")
P("")
P("PROVENANCE: " + "; ".join([f"{SD}/SCORES.txt", f"{SD}/reg/<clip>.json", f"{SD}/reg_wide/0125.json (if 0125)",
                               f"{SD}/decomp/decomp.json + DECOMP.txt", f"{SD}/temporal/temporal.json",
                               "outputs/deep_20261004/sdedit/clips/*/speed_log.json"]))
open(OUTT, "w").write("\n".join(out) + "\n")
json.dump(dict(stage=SD, rule=RULE, clips=clips, results=res,
               k5=dict(n=len(k5), failed=[list(map(str, r)) for r in k5_fail]),
               k6=dict(rows=[[c, a, b] for c, a, b in k6], failed=[list(map(str, r)) for r in k6_fail]),
               k5_decomp=dk5), open(OUTJ, "w"), indent=1, default=float)
