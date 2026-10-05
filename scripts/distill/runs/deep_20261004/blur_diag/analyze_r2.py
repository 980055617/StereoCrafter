#!/usr/bin/env python
"""blur_diag (deep_20261004) -- assemble the decomposition table and the pre-registered readings R1-R4 from the
per-clip JSONs (CPU only).  Definitions: PREREG.txt.
usage: python analyze_r2.py <scores_root> <out_txt> <out_json> [clips...]
  scores_root/{lpips,detail,nr}/<clip>.json
"""
import json
import math
import os
import sys

import numpy as np

ROOT, OUT_TXT, OUT_JSON = sys.argv[1], sys.argv[2], sys.argv[3]
ALL = "0042 0052 0125 0128 0141 0147 0170 0204 0225 0251 0259 0301".split()
CLIPS = sys.argv[4:] or ALL
F4400 = [c for c in CLIPS if c in ALL[:6]]
F2160 = [c for c in CLIPS if c in ALL[6:]]
ORDER = ["GT", "LEFT", "VAE_GT", "VAE_GT32", "VAE_GTx", "VAE_GTx_L", "RS_GTx", "RS_GTx_L", "BR", "VAE_BR", "ORIGIN",
         "DELIV", "S25", "T5NAT", "T5PAD", "HIRES_A", "HIRES_B", "HIRES_B_L"]
GTGEO = ("GT", "LEFT", "VAE_GT", "VAE_GT32", "VAE_GTx", "VAE_GTx_L", "RS_GTx", "RS_GTx_L")
NRP = ["musiq", "clipiqa", "niqe"]
NRX = ["topiq_nr", "clipiqa+", "arniqa"]
L = []


def P(s=""):
    L.append(s)
    print(s)


def load(kind, c):
    p = f"{ROOT}/{kind}/{c}.json"
    return json.load(open(p)) if os.path.exists(p) else None


LP = {c: load("lpips", c) for c in CLIPS}
DT = {c: load("detail", c) for c in CLIPS}
NR = {c: load("nr", c) for c in CLIPS}
missing = [(k, c) for k, D in (("lpips", LP), ("detail", DT), ("nr", NR)) for c in CLIPS if D[c] is None]
# PREREG ADDENDUM 5: merge add-on rows (VAE_GTx_L, RS_GTx_L for 0204) scored by score_addon_r2.py
ADDON = {}
for c in CLIPS:
    a = load("addon", c)
    if a is None:
        continue
    for row, d in a["rows"].items():
        ADDON.setdefault(c, []).append(row)
        if LP[c] is not None and row not in LP[c]["rows"]:
            LP[c]["rows"][row] = dict(path=d["path"], clip=d["lpips"]["clip"])
        if DT[c] is not None and row not in DT[c]["rows"]:
            DT[c]["rows"][row] = d["detail"]
        if NR[c] is not None and row not in NR[c]["rows"]:
            NR[c]["rows"][row] = dict(clip=d["nr"]["clip"])
P("=" * 118)
P("BLUR DIAG -- where does origin's blur come from?   deep_20261004 / blur_diag   (PREREG.txt; sources: "
  f"{ROOT}/{{lpips,detail,nr}}/<clip>.json)")
P("=" * 118)
if missing:
    P(f"MISSING per-clip files: {missing}")


# ------------------------------------------------------------------------------------------------ value getters
def v_lp(c, row, var):
    d = LP[c]
    if d is None or row not in d["rows"] or var not in d["rows"][row]["clip"]:
        return None
    return d["rows"][row]["clip"][var]


def v_dt(c, row, chain, key):
    d = DT[c]
    if d is None or row not in d["rows"] or chain not in d["rows"][row]:
        return None
    return d["rows"][row][chain][key]


def r_dt(c, row, chain, key, ref):
    a, b = v_dt(c, row, chain, key), v_dt(c, ref, chain, key)
    return None if a is None or b is None else a / b


def v_nr(c, row, m):
    d = NR[c]
    if d is None or row not in d["rows"]:
        return None
    return d["rows"][row]["clip"][m]


def mean(vals):
    v = [x for x in vals if x is not None]
    return (float(np.mean(v)), len(v)) if v else (None, 0)


def fmt(x, n=4, w=8):
    return f"{'--':>{w}s}" if x is None else f"{x:{w}.{n}f}"


# ------------------------------------------------------------------------------------------------ gates
P("\nGATES")
g0 = [(c, LP[c]["G0"]["pass_"]) for c in CLIPS if LP[c]]
P(f"  G0 scorer reproduction (ORIGIN/DELIV/S25 UNREG+REG within 1e-4 of eval_robustness): "
  f"{'PASS' if g0 and all(x for _, x in g0) else 'FAIL'} on {sum(x for _, x in g0)}/{len(g0)} clips")
for c in CLIPS:
    if LP[c]:
        g = LP[c]["G0"]["rows"]
        P("    " + c + "  " + "  ".join(f"{r} dU {g[r]['d_unreg']:+.1e} dR {g[r]['d_reg']:+.1e}" for r in g)
          + f"  left-halves identical {LP[c]['left_identical']}")
for fmtn, cl in (("ALL", CLIPS), ("4400", F4400), ("2160", F2160)):
    if not cl:
        continue
    s = "  ".join(f"{r} UNREG {fmt(mean([v_lp(c, r, 'UNREG') for c in cl])[0])} REG {fmt(mean([v_lp(c, r, 'REG') for c in cl])[0])}"
                  for r in ("ORIGIN", "DELIV", "S25"))
    P(f"  {fmtn:4s} means: {s}")


# ------------------------------------------------------------------------------------------------ decomposition table
def table(cl, title):
    P("\n" + "-" * 118)
    P(f"DECOMPOSITION -- {title}  (n clips = {len(cl)}; HIRES_A/B only on the clips that have them, n shown)")
    P("-" * 118)
    P("  LPIPS: REG = vs registered GT (VAE_GT*/RS_GTx vs their own input); UNREG = published geometry; VALID_GT / "
      "VALID_BR = composite, valid pixels")
    P("  GT chain ratios row/GT on valid pixels (edgeHF/edgeGy at GT edges: biased LOW for render-geometry rows); "
      "BR chain = registration-free vs input")
    hdr = (f"  {'row':9s} {'n':>2s} {'REG':>7s} {'UNREG':>7s} {'VAL_GT':>7s} {'VAL_BR':>7s} | {'edgeHF':>6s} {'edgeGy':>6s} "
           f"{'flatHF':>6s} {'stripE':>6s} {'b1':>6s} {'b2':>6s} {'b3':>6s} {'b4':>6s} | {'eHF@BR':>6s} {'b1@BR':>6s} "
           f"{'b2@BR':>6s} | {'sharp':>6s} | {'MUSIQ':>6s} {'CLIPIQA':>7s} {'NIQE':>6s}")
    P(hdr)
    out = {}
    for r in ORDER:
        n = sum(1 for c in cl if (LP[c] and r in LP[c]["rows"]) or (DT[c] and r in DT[c]["rows"]))
        if n == 0:
            continue
        row = dict(
            n=n,
            REG=mean([v_lp(c, r, "REG") for c in cl])[0], UNREG=mean([v_lp(c, r, "UNREG") for c in cl])[0],
            VALID_GT=mean([v_lp(c, r, "VALID_GT") for c in cl])[0], VALID_BR=mean([v_lp(c, r, "VALID_BR") for c in cl])[0],
            **{f"gt_{k}": mean([r_dt(c, r, "gtchain", k, "GT") for c in cl])[0]
               for k in ("edgeHF", "edgeGy", "flatHF", "stripeE", "b1", "b2", "b3", "b4")},
            **{f"br_{k}": mean([r_dt(c, r, "brchain", k, "BR") if r != "BR" else 1.0 for c in cl])[0]
               for k in ("edgeHF", "b1", "b2")} if r not in GTGEO else {},
            sharp=mean([(v_dt(c, r, "raw", "sharp") / v_dt(c, "GT", "raw", "sharp")) if v_dt(c, r, "raw", "sharp") else None
                        for c in cl])[0],
            **{m: mean([v_nr(c, r, m) for c in cl])[0] for m in NRP + NRX},
        )
        if r == "LEFT":
            for k in ("b1", "b2", "b3", "b4"):
                row[f"gt_{k}"] = mean([(v_dt(c, "LEFT", "raw", f"ff_{k}") / v_dt(c, "GT", "raw", f"ff_{k}"))
                                       if DT[c] else None for c in cl])[0]
        out[r] = row
        P(f"  {r:9s} {n:2d} {fmt(row['REG'], 4, 7)} {fmt(row['UNREG'], 4, 7)} {fmt(row['VALID_GT'], 4, 7)} "
          f"{fmt(row['VALID_BR'], 4, 7)} | {fmt(row['gt_edgeHF'], 3, 6)} {fmt(row['gt_edgeGy'], 3, 6)} "
          f"{fmt(row['gt_flatHF'], 3, 6)} {fmt(row['gt_stripeE'], 3, 6)} {fmt(row['gt_b1'], 3, 6)} {fmt(row['gt_b2'], 3, 6)} "
          f"{fmt(row['gt_b3'], 3, 6)} {fmt(row['gt_b4'], 3, 6)} | {fmt(row.get('br_edgeHF'), 3, 6)} "
          f"{fmt(row.get('br_b1'), 3, 6)} {fmt(row.get('br_b2'), 3, 6)} | {fmt(row['sharp'], 3, 6)} | "
          f"{fmt(row['musiq'], 2, 6)} {fmt(row['clipiqa'], 4, 7)} {fmt(row['niqe'], 3, 6)}")
    P("  (LEFT: b1..b4 are full-frame raw band ratios LEFT/GT -- camera/anchor check; BR row: hole-contaminated in REG/UNREG/"
      "sharp/NR)")
    return out


DEC = {"ALL": table(CLIPS, "ALL clips")}
if F4400:
    DEC["4400"] = table(F4400, "4400-px clips (0042 0052 0125 0128 0141 0147)")
if F2160:
    DEC["2160"] = table(F2160, "2160-px clips (0170 0204 0225 0251 0259 0301)")

# ------------------------------------------------------------------------------------------------ contrast-normalised
P("\n" + "-" * 118)
P("DESCRIPTIVE: contrast-normalised fine bands (b_k/GT divided by b4/GT; separates blur from a global contrast change)")
P("-" * 118)
for nm in [k for k in ("ALL", "4400", "2160") if k in DEC]:
    s = "  ".join(f"{r} {fmt((v['gt_b1'] / v['gt_b4']) if v.get('gt_b1') and v.get('gt_b4') else None, 3, 5)}/"
                  f"{fmt((v['gt_b2'] / v['gt_b4']) if v.get('gt_b2') and v.get('gt_b4') else None, 3, 5)}"
                  for r, v in DEC[nm].items() if r not in ("GT", "LEFT"))
    P(f"  {nm:4s} (b1/b4, b2/b4): {s}")

# ------------------------------------------------------------------------------------------------ seed yardstick
P("\n" + "-" * 118)
P("SEED YARDSTICK (proxy): Y_m = max over clips |m(T5NAT) - m(T5PAD)|; mean |diff| in brackets")
P("-" * 118)
YS = {}


def ymetric(name, fn):
    d = [fn(c, "T5NAT") - fn(c, "T5PAD") for c in CLIPS if fn(c, "T5NAT") is not None and fn(c, "T5PAD") is not None]
    if d:
        YS[name] = dict(max=float(np.max(np.abs(d))), mean=float(np.mean(np.abs(d))), n=len(d))
        P(f"  {name:22s} Y {YS[name]['max']:.5f}  [{YS[name]['mean']:.5f}]  n={len(d)}")


ymetric("LPIPS_REG", lambda c, r: v_lp(c, r, "REG"))
ymetric("LPIPS_UNREG", lambda c, r: v_lp(c, r, "UNREG"))
ymetric("LPIPS_VALID_BR", lambda c, r: v_lp(c, r, "VALID_BR"))
ymetric("edgeHF@BR_ratio", lambda c, r: r_dt(c, r, "brchain", "edgeHF", "BR"))
for k in ("b1", "b2"):
    ymetric(f"{k}/GT", lambda c, r, k=k: r_dt(c, r, "gtchain", k, "GT"))
ymetric("edgeHF/GT", lambda c, r: r_dt(c, r, "gtchain", "edgeHF", "GT"))
ymetric("flatHF/GT", lambda c, r: r_dt(c, r, "gtchain", "flatHF", "GT"))
ymetric("stripeE/GT", lambda c, r: r_dt(c, r, "gtchain", "stripeE", "GT"))
ymetric("flatHF@BR_ratio", lambda c, r: r_dt(c, r, "brchain", "flatHF", "BR"))
for m in NRP + NRX:
    ymetric(m, lambda c, r, m=m: v_nr(c, r, m))

# ------------------------------------------------------------------------------------------------ R1 stages
P("\n" + "-" * 118)
P("R1  STAGE RETENTION (PRIMARY = registration-free input chain).  retention < 1 = detail lost in that stage")
P("-" * 118)
STG = {}


def stage_vals(c):
    o = {}
    for X in ("edgeHF", "edgeGy", "b1", "b2", "flatHF", "stripeE"):
        o[f"VAEin_{X}"] = r_dt(c, "VAE_BR", "brchain", X, "BR")
        a, b = v_dt(c, "ORIGIN", "brchain", X), v_dt(c, "VAE_BR", "brchain", X)
        o[f"UNet_{X}"] = None if a is None or b is None else a / b
        a, b = v_dt(c, "DELIV", "brchain", X), v_dt(c, "ORIGIN", "brchain", X)
        o[f"deliv/orig_{X}"] = None if a is None or b is None else a / b
        a, b = v_dt(c, "S25", "brchain", X), v_dt(c, "ORIGIN", "brchain", X)
        o[f"s25/orig_{X}"] = None if a is None or b is None else a / b
        o[f"VAEgt_{X}"] = r_dt(c, "VAE_GT", "gtchain", X, "GT")
    for X in ("b1", "b2", "b3", "b4"):
        o[f"view_{X}"] = r_dt(c, "BR", "gtchain", X, "GT")
        o[f"left/gt_{X}"] = (v_dt(c, "LEFT", "raw", f"ff_{X}") / v_dt(c, "GT", "raw", f"ff_{X}")) if DT[c] else None
    return o


_kc = next((c for c in CLIPS if DT[c]), None)
keys = list(stage_vals(_kc).keys()) if _kc else []
P(f"  {'clip':6s} " + " ".join(f"{k:>13s}" for k in keys))
per = {c: stage_vals(c) for c in CLIPS if DT[c]}
for c, o in per.items():
    P(f"  {c:6s} " + " ".join(fmt(o[k], 3, 13) for k in keys))
for nm, cl in (("ALL", CLIPS), ("4400", F4400), ("2160", F2160)):
    cl = [c for c in cl if c in per]
    if cl:
        STG[nm] = {k: mean([per[c][k] for c in cl])[0] for k in keys}
        P(f"  {nm:6s} " + " ".join(fmt(STG[nm][k], 3, 13) for k in keys))
if "ALL" in STG:
    s = STG["ALL"]
    verdict = {}
    for X in ("edgeHF", "b1"):
        cand = {"VAE on the input (BR->VAE_BR)": s[f"VAEin_{X}"], "UNet generation (VAE_BR->ORIGIN)": s[f"UNet_{X}"]}
        if X == "b1" and s["view_b1"] is not None and s["view_b1"] < 1 and (s.get("left/gt_b1") or 0) >= 1:
            cand["view change + splatting (GT->BR)"] = s["view_b1"]      # PREREG R1 condition
        cand = {k: v for k, v in cand.items() if v is not None}
        verdict[X] = min(cand, key=cand.get) if cand else None
        P(f"  smallest retention by {X}: {verdict[X]}  ({', '.join(f'{k} {v:.3f}' for k, v in cand.items())})")
    P(f"  R1 VERDICT: " + (f"'{verdict['edgeHF']}' loses most detail (edgeHF and b1 agree)"
                          if verdict.get("edgeHF") == verdict.get("b1") else
                          f"edgeHF names '{verdict.get('edgeHF')}', b1 names '{verdict.get('b1')}' -> no single stage named"))
    lb = s.get("left/gt_b1")
    P(f"  view-stage condition: BR/GT b1 {fmt(s.get('view_b1'), 3, 5)}, LEFT/GT b1 {fmt(lb, 3, 5)} "
      f"(view counts as a detail loss only if BR/GT < 1 and LEFT/GT >= 1)")
    STG["verdict"] = verdict
    # descriptive: share of the total log loss BR -> ORIGIN taken by each stage (per clip, then mean), and the
    # share of the UNet stage's loss that DELIV / S25 recover
    P("  DESCRIPTIVE log-loss shares, per clip then mean (total = ln X(ORIGIN)/X(BR) = ln r_VAEin + ln r_UNet):")
    SH = {}
    for X in ("edgeHF", "edgeGy", "b1", "b2"):
        vs, us, dr, sr = [], [], [], []
        for c, o in per.items():
            a, b = o[f"VAEin_{X}"], o[f"UNet_{X}"]
            if a and b and a > 0 and b > 0 and math.log(a) + math.log(b) < 0:
                tot = math.log(a) + math.log(b)
                vs.append(math.log(a) / tot)
                us.append(math.log(b) / tot)
                if o[f"deliv/orig_{X}"] and b < 1:
                    dr.append(math.log(o[f"deliv/orig_{X}"]) / -math.log(b))
                if o[f"s25/orig_{X}"] and b < 1:
                    sr.append(math.log(o[f"s25/orig_{X}"]) / -math.log(b))
        SH[X] = dict(vae_share=mean(vs)[0], unet_share=mean(us)[0], deliv_recovers=mean(dr)[0], s25_recovers=mean(sr)[0],
                     n=len(vs))
        P(f"    {X:7s} VAE-on-input share {fmt(SH[X]['vae_share'], 3, 5)}  UNet share {fmt(SH[X]['unet_share'], 3, 5)}"
          f"  | of the UNet-stage loss, DELIV recovers {fmt(SH[X]['deliv_recovers'], 3, 5)}, S25 recovers "
          f"{fmt(SH[X]['s25_recovers'], 3, 5)}  (n={len(vs)})")
    STG["log_shares"] = SH
    # advisor follow-up (descriptive, same rule per format; the ALL verdict above stays the verdict of record)
    P("  PER-FORMAT verdicts (same R1 rule) and per-clip counts of the stage with the smaller retention:")
    STG["per_format"] = {}
    for nm in ("4400", "2160"):
        if nm not in STG:
            continue
        sv = STG[nm]
        vv = {}
        for X in ("edgeHF", "b1"):
            cand = {"VAE-on-input": sv[f"VAEin_{X}"], "UNet": sv[f"UNet_{X}"]}
            if X == "b1" and sv["view_b1"] is not None and sv["view_b1"] < 1 and (sv.get("left/gt_b1") or 0) >= 1:
                cand["view"] = sv["view_b1"]
            cand = {k: v for k, v in cand.items() if v is not None}
            vv[X] = min(cand, key=cand.get) if cand else None
        cl = [c for c in (F4400 if nm == "4400" else F2160) if c in per]
        cnt = {X: dict(VAE=sum(1 for c in cl if per[c][f"VAEin_{X}"] is not None and per[c][f"UNet_{X}"] is not None
                               and per[c][f"VAEin_{X}"] < per[c][f"UNet_{X}"]),
                       UNet=sum(1 for c in cl if per[c][f"VAEin_{X}"] is not None and per[c][f"UNet_{X}"] is not None
                                and per[c][f"UNet_{X}"] < per[c][f"VAEin_{X}"]))
               for X in ("edgeHF", "edgeGy", "b1", "b2")}
        STG["per_format"][nm] = dict(verdict=vv, counts=cnt)
        P(f"    {nm}: edgeHF -> {vv['edgeHF']}, b1 -> {vv['b1']}  "
          f"({'agree' if vv['edgeHF'] == vv['b1'] else 'DISAGREE'}); clips where the VAE stage / the UNet stage loses more: "
          + "  ".join(f"{X} {cnt[X]['VAE']}/{cnt[X]['UNet']}" for X in cnt))
    cl = [c for c in CLIPS if c in per]
    cnt = {X: (sum(1 for c in cl if per[c][f"VAEin_{X}"] is not None and per[c][f"UNet_{X}"] is not None
                   and per[c][f"VAEin_{X}"] < per[c][f"UNet_{X}"]),
               sum(1 for c in cl if per[c][f"VAEin_{X}"] is not None and per[c][f"UNet_{X}"] is not None
                   and per[c][f"UNet_{X}"] < per[c][f"VAEin_{X}"])) for X in ("edgeHF", "edgeGy", "b1", "b2")}
    STG["counts_all"] = cnt
    P("    ALL : clips where the VAE stage / the UNet stage loses more: " + "  ".join(f"{X} {a}/{b}" for X, (a, b) in cnt.items()))
    P("  VAE on the INPUT vs VAE on the GT (stripe-free comparator; VAEin < VAEgt => part of the input-VAE 'loss' is "
      "artefact removal):")
    for nm in [k for k in ("ALL", "4400", "2160") if k in STG]:
        sv = STG[nm]
        P(f"    {nm:4s} " + "  ".join(f"{X}: VAEin {fmt(sv[f'VAEin_{X}'], 3, 5)} VAEgt {fmt(sv[f'VAEgt_{X}'], 3, 5)}"
                                 for X in ("edgeHF", "edgeGy", "b1", "b2"))
          + f"  | input artefacts through the VAE: flatHF {fmt(sv['VAEin_flatHF'], 3, 5)} stripeE {fmt(sv['VAEin_stripeE'], 3, 5)}"
          + f"; through the UNet: flatHF {fmt(sv['UNet_flatHF'], 3, 5)} stripeE {fmt(sv['UNet_stripeE'], 3, 5)}")

    # PREREG ADDENDUM 7: selection-symmetric edge measures (scores_root/selfedge/<clip>.json)
    SE = {c: load("selfedge", c) for c in CLIPS}
    if any(SE.values()):
        P("  ADDENDUM 7 selection check -- stage retentions with selection-symmetric edge measures (BR chain; VAEgt in GT chain):")
        SEL = {}

        def sev(c, row, chain, key):
            d = SE[c]
            return None if d is None or row not in d["rows"] or chain not in d["rows"][row] else d["rows"][row][chain][key]

        for key in ("edgeHF_self", "edgeHF_s1"):
            pc = {}
            for c in CLIPS:
                br, vb, og = sev(c, "BR", "brchain", key), sev(c, "VAE_BR", "brchain", key), sev(c, "ORIGIN", "brchain", key)
                gt, vg = sev(c, "GT", "gtchain", key), sev(c, "VAE_GT", "gtchain", key)
                if None in (br, vb, og):
                    continue
                pc[c] = dict(VAEin=vb / br, UNet=og / vb, VAEgt=(vg / gt) if gt and vg else None)
            SEL[key] = {}
            for nm, cl in (("ALL", CLIPS), ("4400", F4400), ("2160", F2160)):
                cl = [c for c in cl if c in pc]
                if not cl:
                    continue
                mv = {k: mean([pc[c][k] for c in cl])[0] for k in ("VAEin", "UNet", "VAEgt")}
                stage = None if mv["VAEin"] is None or mv["UNet"] is None else (
                    "VAE on the input (BR->VAE_BR)" if mv["VAEin"] < mv["UNet"] else "UNet generation (VAE_BR->ORIGIN)")
                nv = sum(1 for c in cl if pc[c]["VAEin"] < pc[c]["UNet"])
                SEL[key][nm] = dict(means=mv, stage=stage, clips_vae_loses_more=nv, n=len(cl))
                P(f"    {key:11s} {nm:4s} VAEin {fmt(mv['VAEin'], 3, 5)}  UNet {fmt(mv['UNet'], 3, 5)}  VAEgt "
                  f"{fmt(mv['VAEgt'], 3, 5)} -> larger loss: {stage}  (VAE loses more on {nv}/{len(cl)} clips)")
        ve = STG.get("verdict", {}).get("edgeHF")
        changed = all(SEL[k].get("ALL", {}).get("stage") not in (None, ve) for k in SEL)
        P(f"    edgeHF verdict '{ve}' {'CHANGES under both selection-symmetric measures -> selection-sensitive, the b1 reading carries the conclusion' if changed else 'is NOT overturned by both selection-symmetric measures'}")
        STG["selection_check"] = dict(SEL=SEL, edge_verdict_selection_sensitive=changed)

# ------------------------------------------------------------------------------------------------ R2 VAE ceiling
P("\n" + "-" * 118)
P("R2  VAE CEILING vs ORIGIN (registered LPIPS)")
P("-" * 118)
R2 = {}
for nm, cl in (("ALL", CLIPS), ("4400", F4400), ("2160", F2160)):
    if not cl:
        continue
    vg, og = mean([v_lp(c, "VAE_GT", "REG") for c in cl])[0], mean([v_lp(c, "ORIGIN", "REG") for c in cl])[0]
    vg32, vgx = mean([v_lp(c, "VAE_GT32", "REG") for c in cl])[0], mean([v_lp(c, "VAE_GTx", "REG") for c in cl])[0]
    # paired subsets (clips that have the sensitivity row) so that the comparisons are like-for-like
    c32 = [c for c in cl if v_lp(c, "VAE_GT32", "REG") is not None]
    cxl = [c for c in cl if v_lp(c, "VAE_GTx_L", "REG") is not None]
    P(f"  {nm:4s} paired: VAE_GT vs VAE_GT32 on {c32}: {fmt(mean([v_lp(c, 'VAE_GT', 'REG') for c in c32])[0])} vs "
      f"{fmt(mean([v_lp(c, 'VAE_GT32', 'REG') for c in c32])[0])} | VAE_GT / VAE_GTx / VAE_GTx_L / RS_GTx / RS_GTx_L on {cxl}: "
      + " / ".join(fmt(mean([v_lp(c, r, 'REG') for c in cxl])[0]) for r in ("VAE_GT", "VAE_GTx", "VAE_GTx_L", "RS_GTx", "RS_GTx_L"))
      + " | b1/GT: " + " / ".join(fmt(mean([r_dt(c, r, 'gtchain', 'b1', 'GT') for c in cxl])[0], 3, 5)
                                 for r in ("VAE_GT", "VAE_GTx", "VAE_GTx_L", "RS_GTx", "RS_GTx_L"))
      + " | edgeHF/GT: " + " / ".join(fmt(mean([r_dt(c, r, 'gtchain', 'edgeHF', 'GT') for c in cxl])[0], 3, 5)
                                     for r in ("VAE_GT", "VAE_GTx", "VAE_GTx_L", "RS_GTx", "RS_GTx_L")))
    rsx = mean([v_lp(c, "RS_GTx", "REG") for c in cl])[0]
    vbr = mean([v_lp(c, "VAE_BR", "VALID_BR") for c in cl])[0]
    obr = mean([v_lp(c, "ORIGIN", "VALID_BR") for c in cl])[0]
    R2[nm] = dict(VAE_GT=vg, VAE_GT32=vg32, VAE_GTx=vgx, RS_GTx=rsx, ORIGIN=og, share=(vg / og) if vg and og else None,
                  VAE_BR_vs_BR=vbr, ORIGIN_vs_BR=obr)
    P(f"  {nm:4s} LPIPS VAE_GT {fmt(vg)}  VAE_GT32 {fmt(vg32)}  VAE_GTx {fmt(vgx)}  RS_GTx {fmt(rsx)}  | ORIGIN REG {fmt(og)}"
      f"  -> VAE_GT / ORIGIN = {fmt(R2[nm]['share'], 3, 5)}  | input chain: VAE_BR vs BR {fmt(vbr)}  ORIGIN vs BR {fmt(obr)}")

# ------------------------------------------------------------------------------------------------ R3 NR gap
P("\n" + "-" * 118)
P("R3  NO-REFERENCE GAP (clip means; niqe lower = better, others higher = better)")
P("-" * 118)
R3 = {}
for nm, cl in (("ALL", CLIPS), ("4400", F4400), ("2160", F2160)):
    if not cl:
        continue
    R3[nm] = {}
    for m in NRP + NRX:
        g = {r: mean([v_nr(c, r, m) for c in cl])[0] for r in ("GT", "LEFT", "VAE_GT", "VAE_GTx", "BR", "VAE_BR", "ORIGIN",
                                                                "DELIV", "S25")}
        gap = None if g["ORIGIN"] is None or g["GT"] is None else g["ORIGIN"] - g["GT"]
        closed = {r: ((g[r] - g["ORIGIN"]) / (g["GT"] - g["ORIGIN"])) if gap not in (None, 0) and g[r] is not None else None
                  for r in ("DELIV", "S25", "VAE_GT")}
        unfair = None if g["LEFT"] is None or gap is None else abs(g["GT"] - g["LEFT"]) > abs(gap)
        R3[nm][m] = dict(values=g, gap=gap, closed=closed, anchor_unfair=unfair)
        P(f"  {nm:4s} {m:9s} GT {fmt(g['GT'], 4, 8)} LEFT {fmt(g['LEFT'], 4, 8)} VAE_GT {fmt(g['VAE_GT'], 4, 8)} "
          f"VAE_GTx {fmt(g['VAE_GTx'], 4, 8)} BR {fmt(g['BR'], 4, 8)} VAE_BR {fmt(g['VAE_BR'], 4, 8)} ORIGIN {fmt(g['ORIGIN'], 4, 8)} "
          f"DELIV {fmt(g['DELIV'], 4, 8)} S25 {fmt(g['S25'], 4, 8)} | gap(O-GT) {fmt(gap, 4, 8)} closed: DELIV "
          f"{fmt(closed['DELIV'], 2, 5)} S25 {fmt(closed['S25'], 2, 5)} | anchor unfair: {unfair}")

# ------------------------------------------------------------------------------------------------ R4 working resolution
P("\n" + "-" * 118)
P("R4  WORKING RESOLUTION: HIRES_B (1.75x upsampled window, 1024x1792) and HIRES_A (context) vs ORIGIN, per clip")
P("-" * 118)
R4 = {}
MET = [("edgeHF@BR", lambda c, r: r_dt(c, r, "brchain", "edgeHF", "BR"), +1, "edgeHF@BR_ratio"),
       ("b1/GT", lambda c, r: r_dt(c, r, "gtchain", "b1", "GT"), +1, "b1/GT"),
       ("b2/GT", lambda c, r: r_dt(c, r, "gtchain", "b2", "GT"), +1, "b2/GT"),
       ("LPIPS_REG", lambda c, r: v_lp(c, r, "REG"), -1, "LPIPS_REG"),
       ("LPIPS_UNREG", lambda c, r: v_lp(c, r, "UNREG"), -1, "LPIPS_UNREG"),
       ("LPIPS_VALID_BR", lambda c, r: v_lp(c, r, "VALID_BR"), -1, "LPIPS_VALID_BR")] + \
      [(m, (lambda c, r, m=m: v_nr(c, r, m)), (-1 if m == "niqe" else +1), m) for m in NRP + NRX]
for row in ("HIRES_B", "HIRES_B_L", "HIRES_A"):
    cl = [c for c in CLIPS if LP[c] and row in LP[c]["rows"]]
    if not cl:
        P(f"  {row}: no clip scored")
        continue
    R4[row] = {}
    for name, fn, sign, yk in MET:
        d = {c: (fn(c, row) - fn(c, "ORIGIN")) if fn(c, row) is not None and fn(c, "ORIGIN") is not None else None
             for c in cl}
        y = YS.get(yk, {}).get("max")
        better_all = all(v is not None and y is not None and sign * v > y for v in d.values())
        worse_all = all(v is not None and y is not None and -sign * v > y for v in d.values())
        R4[row][name] = dict(delta=d, Y=y, better_all=better_all, worse_all=worse_all,
                             origin={c: fn(c, "ORIGIN") for c in cl}, row={c: fn(c, row) for c in cl})
        P(f"  {row} {name:15s} " + "  ".join(f"{c}: {fmt(fn(c, 'ORIGIN'), 4, 7)} -> {fmt(fn(c, row), 4, 7)} "
                                           f"(d {fmt(v, 4, 7)})" for c, v in d.items())
          + f"   Y {fmt(y, 4, 6)}  {'BETTER>Y on all' if better_all else ('WORSE>Y on all' if worse_all else 'mixed/within Y')}")
    sharper = all(R4[row][k]["better_all"] for k in ("edgeHF@BR", "b1/GT", "b2/GT"))
    blurrier = all(R4[row][k]["worse_all"] for k in ("edgeHF@BR", "b1/GT", "b2/GT"))
    fid = "BETTER FIDELITY" if R4[row]["LPIPS_REG"]["better_all"] else (
        "WORSE FIDELITY" if R4[row]["LPIPS_REG"]["worse_all"] else "NO CLEAR FIDELITY CHANGE")
    R4[row]["verdict"] = dict(sharpness="SHARPER" if sharper else ("BLURRIER" if blurrier else "NO CLEAR CHANGE"),
                              fidelity=fid, clips=cl)
    P(f"  {row} VERDICT ({'PRIMARY' if row == 'HIRES_B' else ('SENSITIVITY R4-L' if row == 'HIRES_B_L' else 'CONTEXT')}): sharpness {R4[row]['verdict']['sharpness']}; {fid}")
    # advisor follow-up: fake-texture check (descriptive) -- artefact measures of the row vs ORIGIN in both chains
    tex = {}
    for c in cl:
        tex[c] = {k: (v_dt(c, row, ch, key) / v_dt(c, ref, ch, key) if v_dt(c, row, ch, key) and v_dt(c, ref, ch, key) else None,
                      v_dt(c, "ORIGIN", ch, key) / v_dt(c, ref, ch, key) if v_dt(c, "ORIGIN", ch, key) and v_dt(c, ref, ch, key) else None)
                  for k, ch, key, ref in (("flatHF/GT", "gtchain", "flatHF", "GT"), ("stripeE/GT", "gtchain", "stripeE", "GT"),
                                          ("flatHF@BR", "brchain", "flatHF", "BR"), ("stripeE@BR", "brchain", "stripeE", "BR"))}
    R4[row]["texture_check"] = tex
    P(f"  {row} texture check (row vs ORIGIN; a sharpness gain with no fidelity gain and a flatHF jump = texture, not detail):")
    for c, t in tex.items():
        P(f"    {c}: " + "  ".join(f"{k} {fmt(a, 3, 5)} (ORIGIN {fmt(b, 3, 5)})" for k, (a, b) in t.items()))
P("  VAE ceiling at 1.75x (all clips): LPIPS VAE_GT {} -> VAE_GTx {} (RS_GTx resampling-only {}); edgeHF/GT {} -> {}; "
  "b1/GT {} -> {}".format(*(fmt(x, 4, 6) for x in (
      DEC["ALL"].get("VAE_GT", {}).get("REG"), DEC["ALL"].get("VAE_GTx", {}).get("REG"), DEC["ALL"].get("RS_GTx", {}).get("REG"),
      DEC["ALL"].get("VAE_GT", {}).get("gt_edgeHF"), DEC["ALL"].get("VAE_GTx", {}).get("gt_edgeHF"),
      DEC["ALL"].get("VAE_GT", {}).get("gt_b1"), DEC["ALL"].get("VAE_GTx", {}).get("gt_b1")))))

json.dump(dict(clips=CLIPS, decomposition=DEC, yardstick=YS, stages=STG, R2=R2, R3=R3, R4=R4,
               gates=dict(G0=g0), missing=missing), open(OUT_JSON, "w"), indent=1, default=str)
open(OUT_TXT, "w").write("\n".join(L) + "\n")
print(f"wrote {OUT_TXT} {OUT_JSON}")
