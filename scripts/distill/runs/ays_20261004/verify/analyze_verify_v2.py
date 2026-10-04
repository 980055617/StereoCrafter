#!/usr/bin/env python
"""ays_20261004 / verify -- analysis (CPU only).  v2 = v1 with ONE fix in part FINAL (KeyError on AYS8_origin_g100).  Definitions: PREREG.txt (same dir).
usage: python analyze_verify_v1.py PART [--dry SCORE_DIR] ...   PART = V1 | V2 | V4 | FINAL
  V1     gate G1 + primary / 3a / S-b from SCORES_V1_<c>.txt                 -> TABLE_V1_UNREG.txt / .json
  V2     gate G2 + per-clip registered deltas C1-C9 (0042 0259)                -> TABLE_V2_REG.txt / .json
  V4     gate T1' + flicker of AYS5 origin / AYS5 deliverable                  -> TABLE_V4_TEMPORAL.txt / .json
  FINAL  the thesis table                                                       -> TABLE_FINAL.txt / .json
  --dry  (V1 only) read ROW lines from another lane's score files instead (code check; writes to the scratchpad)
Every output refuses to overwrite.  All inputs are read-only.
"""
import glob
import json
import math
import os
import re
import sys

import numpy as np
from scipy import stats

os.chdir("/home/kawa/master_project/StereoCrafter")
L = "scripts/distill/runs/ays_20261004/verify"
O = "outputs/ays_20261004/verify"
RB = "outputs/ays_20261004/robust"
A5L = "scripts/distill/runs/ays_20261004/ays5"
CLIPS = "0042 0052 0125 0128 0141 0147 0170 0204 0225 0251 0259 0301".split()
REGIME = ["0052", "0147", "0204", "0301"]
REG2 = ["0042", "0259"]
NBOOT, SEED = 10000, 20261004
ROWRE = re.compile(r"^ROW clip=(\S+) tag=(\S+) dy=(-?\d+) dx=(-?\d+) leftPSNR=(\S+) lpips=(\S+) sharp=(\S+) "
                   r"gtSharp=(\S+) rightPSNR=(\S+) n=(\d+) path=(\S+)\s*$")
VAR3 = ["UNREG", "REG_CLIP", "REG_FRAME"]
VAR6 = ["UNREG", "REG_CLIP", "REG_FRAME", "REG_FRAME_RAW", "BLK_FRAME", "BLK_LOCAL"]


# ------------------------------------------------------------------------------------------------ statistics
def p_sign(d):
    d = [x for x in d if x != 0]
    n, k = len(d), sum(x < 0 for x in d)
    m = min(k, n - k)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(m + 1)) / 2 ** n) if n else 1.0


def p_flip(d):
    d = np.asarray(d, float)
    n = len(d)
    obs = abs(d.mean())
    signs = ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1) * 2 - 1
    return float(np.mean(np.abs((signs * d).mean(1)) >= obs - 1e-15))


def boot(d):
    d = np.asarray(d, float)
    rng = np.random.default_rng(SEED)                       # fresh generator per contrast (ays5 spec)
    m = d[rng.integers(0, len(d), size=(NBOOT, len(d)))].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def tci(d):
    d = np.asarray(d, float)
    h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / math.sqrt(len(d))
    return float(d.mean() - h), float(d.mean() + h)


def contrast(A, B, clips):
    d = [A[c] - B[c] for c in clips]
    lo, hi = boot(d)
    t0, t1 = tci(d)
    i = int(np.argmax(d))
    return dict(n=len(d), per_clip={c: x for c, x in zip(clips, d)}, mean=float(np.mean(d)), neg=sum(x < 0 for x in d),
                boot95=[lo, hi], t95=[t0, t1], p_sign=p_sign(d), p_flip=p_flip(d), worst=(clips[i], d[i]),
                meanA=float(np.mean([A[c] for c in clips])), meanB=float(np.mean([B[c] for c in clips])))


def fmt_c(name, r, digits=5):
    pc = " ".join(f"{c}:{v:+.4f}" for c, v in r["per_clip"].items())
    return (f"{name}  n={r['n']}\n    per clip: {pc}\n"
            f"    mean {r['mean']:+.{digits}f}  negative {r['neg']}/{r['n']}  worst {r['worst'][0]} {r['worst'][1]:+.4f}  "
            f"t95 [{r['t95'][0]:+.5f},{r['t95'][1]:+.5f}]  boot95 [{r['boot95'][0]:+.5f},{r['boot95'][1]:+.5f}]  "
            f"p_sign {r['p_sign']:.5f}  p_flip {r['p_flip']:.5f}   means {r['meanA']:.4f} vs {r['meanB']:.4f}")


# ------------------------------------------------------------------------------------------------ io helpers
def rows_by_path(files):
    out = {}
    for f in files:
        if not os.path.exists(f):
            continue
        for ln in open(f):
            m = ROWRE.match(ln)
            if m:
                c, tag, dy, dx, lp, lpips, sh, gs, rp, n, p = m.groups()
                out.setdefault(p, []).append(dict(file=f, clip=c, dy=int(dy), dx=int(dx), lpips=float(lpips),
                                                  sharp=float(sh), rightPSNR=float(rp), n=int(n)))
    return out


def write_once(path, text):
    assert not os.path.exists(path), f"refusing to overwrite {path}"
    open(path, "w").write(text)


def reg_json(root, c, wide_root=None):
    p = f"{wide_root}/{c}.json" if (wide_root and c == "0125") else f"{root}/{c}.json"
    return json.load(open(p))


# ------------------------------------------------------------------------------------------------ V1
def part_v1(dry=None):
    refs = json.load(open(f"{L}/V1_PUBLISHED_REFS.json"))
    labs = ["origin_ll", "AYS5pad8_origin_g100", "deliv_g100_T5pad", "AYS8_origin_g100"]
    V = {lab: {} for lab in labs}
    S = {lab: {} for lab in labs}
    out, js, void = [], dict(gate={}, contrasts={}), []
    for c in CLIPS:
        if dry:
            files = [f"{A5L}/SCORES_AYS5_S2_{c}.txt", f"{A5L}/SCORES_AYS5_S3_{c}.txt"]
        else:
            files = sorted(glob.glob(f"{L}/SCORES_V1r_{c}.txt")) or [f"{L}/SCORES_V1_{c}.txt"]
            txt = open(files[0]).read()
            assert "SCORE_DONE rc=0" in txt, f"{files[0]} did not finish with rc=0"
        R = rows_by_path(files)
        errs, devs = [], []
        for lab in labs:
            ref = refs[f"{c}/{lab}"]
            hits = R.get(ref["path"], [])
            if not hits:
                errs.append(f"{lab}: no ROW for {ref['path']}")
                continue
            h = hits[-1]
            dv = h["lpips"] - ref["lpips"]
            devs.append(abs(dv))
            if abs(dv) > 2e-6 or (h["dy"], h["dx"], h["n"]) != (ref["dy"], ref["dx"], ref["n"]):
                errs.append(f"{lab}: lpips {h['lpips']:.6f} vs ref {ref['lpips']:.6f} (d {dv:+.1e}); "
                            f"offset/n {(h['dy'], h['dx'], h['n'])} vs {(ref['dy'], ref['dx'], ref['n'])}")
            V[lab][c], S[lab][c] = h["lpips"], h["sharp"]
        ok = not errs
        js["gate"][c] = dict(ok=ok, errs=errs, max_dev=max(devs) if devs else None, file=files[0])
        out.append(f"  G1 {c}: {'PASS' if ok else 'FAIL'}  max |d| {max(devs) if devs else float('nan'):.1e}  [{files[0]}]"
                   + ("" if ok else "  " + "; ".join(errs)))
        if not ok:
            void.append(c)
    valid = [c for c in CLIPS if c not in void]
    P = contrast(V["deliv_g100_T5pad"], V["AYS5pad8_origin_g100"], valid)
    A3 = contrast(V["deliv_g100_T5pad"], V["AYS8_origin_g100"], valid)
    Sb = contrast(V["AYS5pad8_origin_g100"], V["origin_ll"], valid)
    sh = float(np.mean([S["deliv_g100_T5pad"][c] / S["AYS5pad8_origin_g100"][c] for c in valid]))
    conf = P["neg"] >= 10 and P["boot95"][1] < 0 and abs(P["mean"] - (-0.01118)) <= 1e-5 and not void
    js["contrasts"] = dict(primary=P, threeA=A3, Sb=Sb, sharp_ratio_primary=sh, void=void, confirmed=conf,
                           values={lab: V[lab] for lab in labs}, sharp={lab: S[lab] for lab in labs})
    head = [f"ays_20261004 / verify -- V1 UNREGISTERED RE-SCORE ({'DRY RUN on the ays5 lane files' if dry else 'GPU 1, this lane'})",
            "scorer scripts/distill/runs/fulldata_v2/beyond4/score_clip_ll.py UNCHANGED, SCORE_STEP=4, FFV1 renders; t95 = Student-t "
            "(thesis CI); boot95 = percentile, 10000 resamples of clips, default_rng(20261004) (ays5 spec, code check)", "",
            "GATE G1 (each ROW within 2e-6 of the ays5 lane's ROW value for the same file; same offset and n)"]
    body = ["", "=== PRIMARY: deliverable T5@1.00 pad8 - AYS5 origin @1.00 pad8 (5 evaluations each, same per-window noise) ===",
            fmt_c("[deliv_g100_T5pad - AYS5pad8_origin_g100]", P),
            f"    sharpness ratio deliverable/AYS5 (mean of per-clip ratios) {sh:.3f}",
            f"  VERDICT: negative {P['neg']}/{P['n']} (need >= 10), boot95 upper {P['boot95'][1]:+.5f} (need < 0), mean {P['mean']:+.5f} vs "
            f"the ays5 lane's -0.01118 (|d| {abs(P['mean'] + 0.01118):.1e}, need <= 1e-5), VOID {void or 'none'}  ->  "
            f"{'CONFIRMED' if conf else 'NOT CONFIRMED'}",
            "", "--- secondary (descriptive; must reproduce the ays5 tables) ---",
            fmt_c("3a  [deliv_g100_T5pad (5 evals) - AYS8_origin_g100 (8 evals)]   ays5: -0.00125, 7/12", A3),
            fmt_c("S-b [AYS5pad8_origin_g100 (5 evals) - origin_ll (16 evals)]     ays5: -0.00025, 8/12", Sb),
            "", "12-clip means: " + "  ".join(f"{lab} {np.mean([V[lab][c] for c in valid]):.4f}" for lab in labs)]
    text = "\n".join(head + out + body) + "\n"
    if dry:
        print(text)
        return
    write_once(f"{L}/TABLE_V1_UNREG.txt", text)
    write_once(f"{L}/TABLE_V1_UNREG.json", json.dumps(js, indent=1))
    print(text)


# ------------------------------------------------------------------------------------------------ V2
CONTRASTS = [("C1", "mstudent2_step800_deliv_ll", "AYS8_origin_g101", "deliverable 8x2 - AYS8 origin @1.01 (16 v 16)"),
             ("C2", "deliv_g100_T5pad", "origin_g100_T5pad", "deliverable T5pad - origin Karras-T5pad (5 v 5)"),
             ("C3", "AYS8_origin_g101", "origin_ll", "AYS8 origin @1.01 - Karras origin (16 v 16)"),
             ("C4", "mstudent2_step800_deliv_ll", "origin_ll", "deliverable 8x2 - Karras origin (16 v 16)"),
             ("C5", "deliv_g100_T5pad", "AYS8_origin_g101", "deliverable T5pad - AYS8 origin @1.01 (5 v 16)"),
             ("C6", "deliv_g100_T5pad", "AYS5pad8_origin_g100", "deliverable T5pad - AYS5 origin (5 v 5)"),
             ("C6b", "AYS5pad8_origin_g100", "origin_g100_T5pad", "AYS5 origin - origin Karras-T5pad (5 v 5)"),
             ("C7", "s25_ll", "AYS8_origin_g101", "s25 - AYS8 origin @1.01 (50 v 16)"),
             ("C8", "deliv_g100_T5nat", "AYS8_origin_g101", "deliverable T5nat (ships) - AYS8 origin @1.01 (5 v 16)"),
             ("C9", "deliv_g100_T5pad", "AYS8_origin_g100", "NEW: deliverable T5pad - AYS8 origin @1.00 (5 v 8) = 3a registered")]


def robust_cfg(c):
    """label -> config dict from the robust lane's JSONs (pass 1 + AYS5 pass); 0125 from the widened files."""
    a = reg_json(f"{RB}/score_reg_v1", c, f"{RB}/score_reg_v1_wide")
    b = reg_json(f"{RB}/score_reg_ays5_v1", c, f"{RB}/score_reg_ays5_v1_wide")
    cfg = dict(a["configs"])
    cfg.update({k: v for k, v in b["configs"].items() if k not in cfg})
    return cfg, a["reg"], b["reg"], b["configs"]


def part_v2():
    rows = json.load(open(f"{L}/ROWS_reg_v1.json"))
    out, js = [], dict(gate={}, deltas={})
    for c in REG2:
        mine = json.load(open(f"{O}/score_reg_v1/{c}.json"))
        mc = mine["configs"]
        rcfg, rreg1, rreg5, r5cfg = robust_cfg(c)
        errs, maxdev = [], 0.0
        for k in ("clip_ddy", "clip_ddx", "smooth_ddy", "smooth_ddx", "raw_ddy", "raw_ddx"):
            for rr, nm in ((rreg1, "pass1"), (rreg5, "ays5pass")):
                if mine["reg"][k] != rr[k]:
                    errs.append(f"registration {k} differs from robust {nm}")
        lefts = {mc[l]["md5_left"] for l in rows["labels"]}
        if len(lefts) != 1:
            errs.append(f"md5_left differs across my rows ({len(lefts)} distinct)")
        for lab in rows["labels"]:
            pub = rows["cells"][c][lab]["lpips"]
            du = mc[lab]["lpips_clip"]["UNREG"] - pub
            if abs(du) > 2e-6:
                errs.append(f"{lab} UNREG {mc[lab]['lpips_clip']['UNREG']:.6f} vs published ROW {pub:.6f}")
            refs = []
            if lab in rcfg:
                refs.append(("robust", rcfg[lab]))
            if lab in r5cfg and lab != "AYS5pad8_origin_g100":
                refs.append(("robust-ays5pass", r5cfg[lab]))
            for nm, r in refs:
                for v in VAR6:
                    dv = abs(mc[lab]["lpips_clip"][v] - r["lpips_clip"][v])
                    maxdev = max(maxdev, dv)
                    if dv > 1e-5:
                        errs.append(f"{lab} {v} {mc[lab]['lpips_clip'][v]:.6f} vs {nm} {r['lpips_clip'][v]:.6f}")
                if mc[lab]["md5_left"] != r["md5_left"]:
                    errs.append(f"{lab} md5_left differs from {nm}")
        ok = not errs
        js["gate"][c] = dict(ok=ok, errs=errs, max_dev=maxdev)
        out.append(f"  G2 {c}: {'PASS' if ok else 'FAIL'}  max |d| vs robust (all variants, shared rows) {maxdev:.1e}; "
                   f"md5_left distinct across my 9 rows: {len(lefts)}" + ("" if ok else "  ERRORS: " + "; ".join(errs)))
        # per-clip deltas, mine vs robust
        for cid, a, b, desc in CONTRASTS:
            for v in VAR3:
                dm = mc[a]["lpips_clip"][v] - mc[b]["lpips_clip"][v]
                dr = (rcfg[a]["lpips_clip"][v] - rcfg[b]["lpips_clip"][v]) if (a in rcfg and b in rcfg) else None
                js["deltas"].setdefault(cid, {}).setdefault(c, {})[v] = dict(mine=dm, robust=dr)
    lines = ["ays_20261004 / verify -- V2 REGISTERED RE-SCORE, clips 0042 0259 (GPU 1, this lane)",
             "scorer scripts/distill/runs/more_20261004/eval_robustness/score_registered_v1.py UNCHANGED, SCORE_STEP=4; mine: "
             f"{O}/score_reg_v1/<clip>.json; reference: {RB}/score_reg_v1/ and score_reg_ays5_v1/ (read only)", "",
             "GATE G2 (6 variants within 1e-5 of the robust lane, identical registration arrays and md5_left; UNREG within 2e-6 of "
             "the published ROW)"] + out + ["", "--- per-clip deltas A - B (mine | robust; same sign?) ---"]
    for cid, a, b, desc in CONTRASTS:
        lines.append(f"  {cid} {desc}")
        for c in REG2:
            cells = []
            for v in VAR3:
                e = js["deltas"][cid][c][v]
                if e["robust"] is None:
                    cells.append(f"{v} {e['mine']:+.4f} | (none)")
                else:
                    same = (e["mine"] < 0) == (e["robust"] < 0)
                    cells.append(f"{v} {e['mine']:+.4f} | {e['robust']:+.4f} {'same' if same else 'DIFF'}")
            lines.append(f"    {c}: " + "   ".join(cells))
    text = "\n".join(lines) + "\n"
    write_once(f"{L}/TABLE_V2_REG.txt", text)
    write_once(f"{L}/TABLE_V2_REG.json", json.dumps(js, indent=1))
    print(text)


# ------------------------------------------------------------------------------------------------ V4
def tkey(c, lab, path_hint=None):
    return f"{c}_{lab}"


def load_temporal(path, c):
    return json.load(open(path))[c]


def part_v4():
    out, js = [], dict(gate={}, rows={})
    W = {}
    for c in CLIPS:
        mine = load_temporal(f"{O}/temporal_v1/{c}.json", c)
        rob = load_temporal(f"{RB}/temporal_v1/{c}.json", c)
        log = open(f"{O}/temporal_v1/{c}.log").read()
        errs, maxdev = [], 0.0
        if "[temporal] RAFT loaded" not in log:
            errs.append("RAFT not loaded")
        for lab in ("origin_ll", "deliv_g100_T5pad"):
            k = f"{c}_{lab}"
            for m in ("warp", "tLP", "seam", "nonseam"):
                dv = abs(mine[k][m] - rob[k][m])
                maxdev = max(maxdev, dv)
                if not (dv <= 2e-5):
                    errs.append(f"{lab} {m} {mine[k][m]} vs robust {rob[k][m]}")
            for m in ("t0", "l0", "n"):
                if mine[k][m] != rob[k][m]:
                    errs.append(f"{lab} {m} {mine[k][m]} vs {rob[k][m]}")
        tags = [k for k in mine if k != "GT"]
        for k in tags:
            if not (isinstance(mine[k].get("warp"), float) and math.isfinite(mine[k]["warp"])):
                errs.append(f"{k} warp not finite")
        js["gate"][c] = dict(ok=not errs, errs=errs, max_dev=maxdev)
        out.append(f"  T1' {c}: {'PASS' if not errs else 'FAIL'}  max |d| {maxdev:.1e}" + ("" if not errs else "  " + "; ".join(errs)))
        W[c] = dict(origin_ll=mine[f"{c}_origin_ll"], T5pad=mine[f"{c}_deliv_g100_T5pad"],
                    AYS5=mine[f"{c}_AYS5pad8_origin_g100"], originT5pad=rob[f"{c}_origin_g100_T5pad"],
                    T5nat=rob[f"{c}_deliv_g100_T5nat"], AYS8=rob[f"{c}_AYS8_origin_g101"],
                    deliv8=rob[f"{c}_mstudent2_step800_deliv_ll"], GT=mine["GT"])
        if c in REGIME:
            W[c]["AYS5deliv"] = mine[f"{c}_AYS5pad8_deliv_g100"]
    js["W"] = W

    def ratio(a, b, m="warp", clips=CLIPS):
        A = np.mean([W[c][a][m] for c in clips])
        B = np.mean([W[c][b][m] for c in clips])
        k = sum(W[c][a][m] > W[c][b][m] for c in clips)
        return A / B - 1, k, len(clips), A, B

    lines = ["ays_20261004 / verify -- V4 FLICKER (score_temporal_ll.py UNCHANGED, every frame, GPU 1, this lane)",
             f"mine: {O}/temporal_v1/<clip>.json; robust (read only): {RB}/temporal_v1/<clip>.json", "",
             "GATE T1' (origin_ll and deliverable T5pad warp/tLP/seam/nonseam within 2e-5 of robust; t0/l0/n equal; RAFT loaded; finite)"] + out
    lines += ["", "--- increase = ratio of 12-clip means - 1 (V1 definition); k = clips where the first row is higher ---"]
    res = {}
    for m in ("warp", "tLP"):
        for a, b, desc in (("AYS5", "origin_ll", "AYS5 origin vs Karras origin 8x2"),
                           ("AYS5", "originT5pad", "AYS5 origin vs origin Karras-T5pad"),
                           ("T5pad", "AYS5", "deliverable T5pad vs AYS5 origin (the 5-evaluation pair)"),
                           ("T5nat", "AYS5", "deliverable T5nat (ships) vs AYS5 origin"),
                           ("T5pad", "origin_ll", "deliverable T5pad vs Karras origin (reproduces robust +21.4 %)")):
            r = ratio(a, b, m)
            res[f"{m}:{a}/{b}"] = r
            lines.append(f"  [{m}] {desc:58s} {r[0]*100:+6.1f} %   higher on {r[1]}/{r[2]}   means {r[3]:.5f} vs {r[4]:.5f}")
        for a, b, desc in (("AYS5deliv", "T5pad", "AYS5-sigma deliverable vs deliverable T5pad (4 regime clips)"),
                           ("AYS5deliv", "origin_ll", "AYS5-sigma deliverable vs Karras origin (4 regime clips)"),
                           ("T5pad", "origin_ll", "deliverable T5pad vs Karras origin (same 4 clips)"),
                           ("AYS5deliv", "AYS5", "AYS5-sigma deliverable vs AYS5 origin (4 regime clips)")):
            r = ratio(a, b, m, REGIME)
            res[f"{m}:{a}/{b}:4"] = r
            lines.append(f"  [{m}] {desc:58s} {r[0]*100:+6.1f} %   higher on {r[1]}/{r[2]}   means {r[3]:.5f} vs {r[4]:.5f}")
    lines.append("  per clip warp ratio AYS5/Karras: " + "  ".join(f"{c}:{W[c]['AYS5']['warp'] / W[c]['origin_ll']['warp']:.3f}" for c in CLIPS))
    lines.append("  per clip warp ratio T5pad/AYS5:  " + "  ".join(f"{c}:{W[c]['T5pad']['warp'] / W[c]['AYS5']['warp']:.3f}" for c in CLIPS))
    lines.append("  PREDICTION (PREREG, b_gen 1.5475 x ln sharpness 0.998): about -0.3 % vs Karras origin")
    js["ratios"] = {k: list(v) for k, v in res.items()}
    text = "\n".join(lines) + "\n"
    write_once(f"{L}/TABLE_V4_TEMPORAL.txt", text)
    write_once(f"{L}/TABLE_V4_TEMPORAL.json", json.dumps(js, indent=1))
    print(text)


# ------------------------------------------------------------------------------------------------ FINAL
def part_final():
    v1 = json.load(open(f"{L}/TABLE_V1_UNREG.json"))["contrasts"]
    v4 = json.load(open(f"{L}/TABLE_V4_TEMPORAL.json")) if os.path.exists(f"{L}/TABLE_V4_TEMPORAL.json") else None
    # registered (REG_FRAME) and UNREG per clip from the robust lane's JSONs (0125 widened), every 12-clip row
    U, G = {}, {}
    for c in CLIPS:
        cfg, _, _, _ = robust_cfg(c)
        for lab, v in cfg.items():
            U.setdefault(lab, {})[c] = v["lpips_clip"]["UNREG"]
            G.setdefault(lab, {})[c] = v["lpips_clip"]["REG_FRAME"]
    # my V1 values replace the UNREG cells of the four rows I re-scored (they must agree within 2e-6 anyway)
    for lab, vals in v1["values"].items():
        for c, x in vals.items():
            if c in U.get(lab, {}) and abs(U[lab][c] - x) > 2e-6:   # v2: v1 raised KeyError for a row absent from the robust JSONs
                raise SystemExit(f"V1 vs robust UNREG disagree {lab} {c}: {x} vs {U[lab][c]}")
            U.setdefault(lab, {})[c] = x
    # AYS5-sigma deliverable (4 clips, UNREG only) from the ays5 lane's S3 ROW lines
    a5d = {}
    for c in REGIME:
        for p, hits in rows_by_path([f"{A5L}/SCORES_AYS5_S3_{c}.txt"]).items():
            if "AYS5pad8_deliv_g100" in p:
                a5d[c] = hits[-1]["lpips"]
    tw = {}
    T = "outputs/ays_20261004/robust/temporal_v1"
    for c in CLIPS:
        r = json.load(open(f"{T}/{c}.json"))[c]
        for lab in ("origin_ll", "AYS8_origin_g101", "mstudent2_step800_deliv_ll", "deliv_g100_T5nat", "deliv_g100_T5pad"):
            tw.setdefault(lab, {})[c] = r[f"{c}_{lab}"]["warp"]
    if v4:
        for c in CLIPS:
            tw.setdefault("AYS5pad8_origin_g100", {})[c] = v4["W"][c]["AYS5"]["warp"]

    def flick(lab, clips=CLIPS):
        if lab not in tw:
            return None
        return np.mean([tw[lab][c] for c in clips]) / np.mean([tw["origin_ll"][c] for c in clips]) - 1

    def m12(D, lab):
        return float(np.mean([D[lab][c] for c in CLIPS])) if lab in D and len(D[lab]) == 12 else None

    pairs = {  # row -> (matching-cost best origin, description)
        "AYS8_origin_g101": ("origin_ll", "Karras origin 8x2"),
        "AYS5pad8_origin_g100": ("origin_g100_T5pad", "Karras-T5 origin 5x1"),
        "mstudent2_step800_deliv_ll": ("AYS8_origin_g101", "AYS8 origin 8x2"),
        "deliv_g100_T5pad": ("AYS5pad8_origin_g100", "AYS5 origin 5x1"),
    }
    rowsdef = [("Karras origin 8x2 @1.01 (deployed)", "origin_ll", "8 x 2 = 16"),
               ("AYS8 origin 8x2 @1.01", "AYS8_origin_g101", "8 x 2 = 16"),
               ("AYS8 origin 8x1 @1.00 (context)", "AYS8_origin_g100", "8 x 1 = 8"),
               ("AYS5 origin 5x1 @1.00", "AYS5pad8_origin_g100", "5 x 1 = 5"),
               ("deliverable 8x2 @1.01", "mstudent2_step800_deliv_ll", "8 x 2 = 16"),
               ("deliverable T5 5x1 @1.00, paired (pad8)", "deliv_g100_T5pad", "5 x 1 = 5"),
               ("deliverable T5 5x1 @1.00, as shipped (unpadded)", "deliv_g100_T5nat", "5 x 1 = 5")]
    js = dict(rows={}, notes=[])
    lines = ["ays_20261004 / verify -- FINAL THESIS TABLE (12 test clips; lossless FFV1; LPIPS-alex vs the real right eye, SCORE_STEP=4)",
             "UNREG: score_clip_ll.py (V1 re-score for Karras/AYS8@1.00/AYS5/T5pad; robust R0-gated values for the rest).  REG_FRAME: "
             "score_registered_v1.py per-frame registered GT (robust lane per-clip JSONs, 0125 widened grid).  'better' = clips where the "
             "row beats its matching-cost best origin; CI = Student-t 95 % of that paired delta (percentile bootstrap in brackets).  "
             "Evaluations = UNet calls x batch per 14-frame window.  Flicker = warp error, ratio of 12-clip means vs Karras origin - 1.", ""]
    hdr = (f"{'row':48s} {'LPIPS':>7s} {'REG':>7s}   {'vs matching-cost best origin':32s} {'d UNREG':>8s} {'better':>6s} "
           f"{'t95 CI (boot95)':>34s} {'d REG':>8s} {'better':>6s} {'t95 CI REG':>20s}  {'evals':>10s} {'flicker':>8s}")
    lines += [hdr, "-" * len(hdr)]
    for name, lab, ev in rowsdef:
        u, g = m12(U, lab), m12(G, lab)
        rec = dict(unreg=u, reg=g, evals=ev, flicker=flick(lab))
        comp = pairs.get(lab)
        if comp and comp[0] in U:
            cu = contrast(U[lab], U[comp[0]], CLIPS)
            cg = contrast(G[lab], G[comp[0]], CLIPS) if (lab in G and comp[0] in G and len(G[lab]) == 12) else None
            rec.update(vs=comp[1], unreg_c=cu, reg_c=cg)
            cmp_s = (f"{comp[1]:32s} {cu['mean']:+8.4f} {cu['neg']:>3d}/12 "
                     f"[{cu['t95'][0]:+.4f},{cu['t95'][1]:+.4f}] ([{cu['boot95'][0]:+.4f},{cu['boot95'][1]:+.4f}]) "
                     + (f"{cg['mean']:+8.4f} {cg['neg']:>3d}/12 [{cg['t95'][0]:+.4f},{cg['t95'][1]:+.4f}]" if cg else f"{'n/a':>8s}"))
        elif lab == "AYS8_origin_g100":
            cu = contrast(U["deliv_g100_T5pad"], U[lab], CLIPS)
            rec.update(vs="(deliverable T5pad minus this row, 5 v 8 evals)", unreg_c=cu)
            cmp_s = f"{'n/a (best 8-eval origin known)':32s} {'':8s} {'':>6s} {'':34s} {'(2 clips only, V2)':>20s}"
        elif lab == "deliv_g100_T5nat":
            cu = contrast(U[lab], U["AYS5pad8_origin_g100"], CLIPS)
            rec.update(vs="AYS5 origin 5x1 (window 0 paired only)", unreg_c=cu)
            cmp_s = (f"{'AYS5 origin (window-0 pairing)':32s} {cu['mean']:+8.4f} {cu['neg']:>3d}/12 "
                     f"[{cu['t95'][0]:+.4f},{cu['t95'][1]:+.4f}] ([{cu['boot95'][0]:+.4f},{cu['boot95'][1]:+.4f}])")
        else:
            cmp_s = f"{'reference':32s}"
        fl = rec["flicker"]
        lines.append(f"{name:48s} {u:7.4f} {(f'{g:7.4f}' if g is not None else '    n/a'):>7s}   {cmp_s:136s}  {ev:>10s} "
                     f"{(f'{fl*100:+7.1f}%' if fl is not None else '    n/m'):>8s}")
        js["rows"][lab] = rec
    # AYS5-sigma deliverable, 4 regime clips
    if len(a5d) == 4:
        d5 = contrast(a5d, {c: U["AYS5pad8_origin_g100"][c] for c in REGIME}, REGIME)
        d5b = contrast(a5d, {c: U["deliv_g100_T5pad"][c] for c in REGIME}, REGIME)
        f4 = None
        if v4:
            f4 = (np.mean([v4["W"][c]["AYS5deliv"]["warp"] for c in REGIME]) /
                  np.mean([v4["W"][c]["origin_ll"]["warp"] for c in REGIME]) - 1)
        lines.append(f"{'AYS5-sigma deliverable 5x1 @1.00 (n=4 ONLY)':48s} {np.mean(list(a5d.values())):7.4f} {'    n/m':>7s}   "
                     f"{'AYS5 origin 5x1 (4 clips)':32s} {d5['mean']:+8.4f} {d5['neg']:>3d}/4  "
                     f"[{d5['t95'][0]:+.4f},{d5['t95'][1]:+.4f}] ([{d5['boot95'][0]:+.4f},{d5['boot95'][1]:+.4f}]) {'':29s}"
                     f"  {'5 x 1 = 5':>10s} {(f'{f4*100:+7.1f}%' if f4 is not None else '    n/m'):>8s} (4-clip)")
        lines.append(f"{'':48s} 4-clip context: deliverable T5pad {np.mean([U['deliv_g100_T5pad'][c] for c in REGIME]):.4f}, "
                     f"AYS5 origin {np.mean([U['AYS5pad8_origin_g100'][c] for c in REGIME]):.4f}, Karras origin "
                     f"{np.mean([U['origin_ll'][c] for c in REGIME]):.4f}; AYS5-sigma minus T5pad {d5b['mean']:+.5f} ({d5b['neg']}/4)")
        js["rows"]["AYS5pad8_deliv_g100"] = dict(unreg4=float(np.mean(list(a5d.values()))), vs_ays5=d5, vs_T5pad=d5b, flicker4=f4)
    lines += ["", "n/a = not applicable; n/m = not measured.  AYS8 8x1 @1.00 flicker n/m (its @1.01 twin: see AYS8 8x2 row).",
              "Sources: TABLE_V1_UNREG.json (this lane), outputs/ays_20261004/robust/score_reg_v1*/ and score_reg_ays5_v1*/ (per-clip "
              "JSONs), outputs/ays_20261004/robust/temporal_v1/, TABLE_V4_TEMPORAL.json (this lane), ays5 SCORES_AYS5_S3_<c>.txt."]
    text = "\n".join(lines) + "\n"
    write_once(f"{L}/TABLE_FINAL.txt", text)
    write_once(f"{L}/TABLE_FINAL.json", json.dumps(js, indent=1, default=float))
    print(text)


if __name__ == "__main__":
    part = sys.argv[1]
    if part == "V1":
        part_v1(dry=("--dry" in sys.argv))
    elif part == "V2":
        part_v2()
    elif part == "V4":
        part_v4()
    elif part == "FINAL":
        part_final()
    else:
        sys.exit(f"bad part {part}")
