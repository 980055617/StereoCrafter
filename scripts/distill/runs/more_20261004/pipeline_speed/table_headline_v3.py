#!/usr/bin/env python
"""[v3 = v2 with F10's verdict spelled out (FAIL reasons) and the like-for-like lines appended from
TABLE_LIKE_FOR_LIKE.txt; v2 printed "FAIL/PENDING".]
[v2 of table_headline_v1.py: explicit tiers + the F10 (compiled VAE decoder) rows with their gate status + harness note]
Headline end-to-end table (pre-registered H rule): wall-clock per config, speedup vs deployed origin, md5 / LPIPS gates.
usage: table_headline_v2.py <out_txt (new)>"""
import io, contextlib, json, os, re, statistics, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_stages_v2 as A

O = "outputs/more_20261004/pipeline_speed"
REF = {"576_deliv": "a41b432baeb25393bb12d39b4f2e6a21", "576_origin": "2e533d7755c950d2fc95043f6fb0a51d"}


def load(d, rep=0):
    if not os.path.exists(os.path.join(d, "stage_log.json")):
        return None
    with contextlib.redirect_stdout(io.StringIO()):
        res = A.main([d])[d]
    rr = [x for x in res["runs"] if x["rep"] == rep]
    if not rr:
        return None
    r = rr[0]
    return dict(ref=r["ref"], md5=r["md5"], ex=r["exclusive"], w=r["windows"], process_s=res["process_s"])


def lpips_rows(path):
    out = {}
    if os.path.exists(path):
        for line in open(path):
            m = re.match(r"ROW clip=(\S+) tag=(\S+) .*lpips=([0-9.]+) sharp=([0-9.]+)", line)
            if m:
                out[m.group(2)] = (float(m.group(3)), float(m.group(4)))
    return out


SC = lpips_rows(f"{O}/p_576/SCORES_P1_vs_ref.txt")
ref_l = SC.get("0301_deliv_g100_T5nat", (None, None))[0]
p1_l = SC.get("0301_P1_deliv_T5_576_idfix_vaecompile_cold", (None, None))[0]
p1 = load(f"{O}/p_576/clips/0301_P1_deliv_T5_576_idfix_vaecompile_cold")
p2 = load(f"{O}/p_576/clips/0301_P2_deliv_T5_576_idfix_vaecompile_warm")
lp_ok = (ref_l is not None and p1_l is not None and abs(ref_l - 0.399416) <= 0.0001 and abs(p1_l - ref_l) <= 0.0005)
det_ok = bool(p1 and p2 and p1["md5"] == p2["md5"])
p4 = load(f"{O}/p_576/clips/0301_P4_deliv_T5_576_idfix_vaecompile_worker")
F10 = ("PASS" if (lp_ok and det_ok) else "FAIL") + \
      f" [LPIPS {'within' if lp_ok else 'OUTSIDE'} 0.0005: ref {ref_l} -> P1 {p1_l}; determinism {'ok' if det_ok else 'FAILED'}: " \
      f"md5 P1 {p1['md5'][:8] if p1 else '-'} / P2 {p2['md5'][:8] if p2 else '-'} / P4 {p4['md5'][:8] if p4 else '-'}] -> NOT ADOPTED"

TIERS = {
    "576": [
        ("BASELINES", None),
        ("deployed origin 8x2@1.01", ("origin", [f"{O}/a_576/clips/0301_A2_origin_s8_576", f"{O}/c_576/clips/0301_A2r_origin_s8_576"], 0)),
        ("deliverable T5@1.00 (as shipped today)", ("deliv", [f"{O}/a_576/clips/0301_A1_deliv_T5_576", f"{O}/c_576/clips/0301_A1r_deliv_T5_576"], 0)),
        ("TIER 1  bit-exact, single-clip CLI process (md5-gated)", None),
        ("deliverable T5 + identity fixes F5-F9", ("deliv", [f"{O}/c_576/clips/0301_C1a_deliv_T5_576_idfix", f"{O}/c_576/clips/0301_C1b_deliv_T5_576_idfix_rep"], 0)),
        ("origin + model-independent fixes F6-F9", ("origin", [f"{O}/c_576/clips/0301_C3_origin_s8_576_idfix"], 0)),
        ("TIER 2  bit-exact, 2nd clip in a persistent process (model reload still inside)", None),
        ("deliverable T5 + F5-F9, rep 1 of C1w", ("deliv", [f"{O}/c_576/clips/0301_C1w_deliv_T5_576_idfix_worker"], 1)),
        ("TIER 3  PIXEL: + torch.compile(vae.decoder) [F10]  gate: " + F10, None),
        ("  cold CLI process (compilation inside)", ("compile", [f"{O}/p_576/clips/0301_P1_deliv_T5_576_idfix_vaecompile_cold"], 0)),
        ("  warm inductor/FX cache, new CLI process", ("compile", [f"{O}/p_576/clips/0301_P2_deliv_T5_576_idfix_vaecompile_warm"], 0)),
        ("  2nd clip in a persistent process (P4 rep 1)", ("compile", [f"{O}/p_576/clips/0301_P4_deliv_T5_576_idfix_vaecompile_worker"], 1)),
    ],
    "1792": [
        ("BASELINES", None),
        ("deployed origin 8x2@1.01", ("origin", [f"{O}/a_1792/clips/0301_A4_origin_s8_1792"], 0)),
        ("deliverable T5@1.00 (as shipped today)", ("deliv", [f"{O}/a_1792/clips/0301_A3_deliv_T5_1792"], 0)),
        ("TIER 1  bit-exact, single-clip CLI process (md5 == A3)", None),
        ("deliverable T5 + identity fixes F5-F9", ("deliv", [f"{O}/c_1792/clips/0301_C2_deliv_T5_1792_idfix"], 0)),
        ("TIER 3  PIXEL: + torch.compile(vae.decoder) [F10, gate judged at 576]", None),
        ("  cold CLI process (compilation inside)", ("compile", [f"{O}/p_1792/clips/0301_P3_deliv_T5_1792_idfix_vaecompile_cold"], 0)),
    ],
}

lines, J = [], {}
for res_name, rows in TIERS.items():
    lines.append(f"===== 0301 @ {'576x1024' if res_name == '576' else '1024x1792'} " + "=" * 90)
    lines.append(f"  {'config':58s} {'n':>2s} {'end-to-end s':>12s} {'(each)':>15s} {'x vs origin':>11s} "
                 f"{'UNet s':>7s} {'VAEdec s':>8s} {'w0 xs':>6s} {'md5':>9s} {'gate':>5s}")
    base = None
    for label, spec in rows:
        if spec is None:
            lines.append(f"  -- {label}")
            continue
        model, dirs, rep = spec
        got = [g for g in (load(d, rep) for d in dirs) if g]
        if not got:
            lines.append(f"  {label:58s}  -  (not rendered)")
            continue
        m = statistics.mean(g["ref"] for g in got)
        if base is None:
            base = m
        md5s = sorted({g["md5"] for g in got})
        key = f"{res_name}_{model}"
        if key not in REF and model != "compile" and len(md5s) == 1:
            REF[key] = md5s[0]          # 1792: the first (baseline) row of each model defines the reference
        if model == "compile":
            gate = "PIX"
        else:
            gate = "PASS" if md5s == [REF.get(key)] else "FAIL"
        each = "/".join(f"{g['ref']:.1f}" for g in got)
        w0x = statistics.mean((g["w"]["first_call_ms"] - g["w"]["second_call_ms"]) / 1000 for g in got)
        lines.append(f"  {label:58s} {len(got):>2d} {m:12.2f} {each:>15s} {base / m:10.3f}x "
                     f"{statistics.mean(g['ex'].get('unet_gpu', 0) for g in got):7.2f} "
                     f"{statistics.mean(g['ex'].get('vae_decode', 0) for g in got):8.2f} {w0x:6.2f} "
                     f"{','.join(x[:8] for x in md5s):>9s} {gate:>5s}")
        J[f"{res_name}|{label}"] = dict(dirs=dirs, rep=rep, end_to_end_each=[g["ref"] for g in got], mean=m,
                                        speedup_vs_origin=base / m, md5=md5s, gate=gate)
    lines.append("")
lines += [
    "end-to-end = python start -> exit of a single-clip CLI process (driver wall-clock inside the GPU-0 lock; nothing else on",
    "  GPU 0), except TIER 2 / persistent rows = that clip's own run time inside a process that already rendered one clip",
    "  (model reload still included; no python start-up/import; no autotune).  'w0 xs' = first UNet forward minus the second.",
    "md5 gate = pre-encode sbs md5 equals the reference render of the same model+resolution (576: the speed-lane references",
    "  a41b432b / 2e533d77; 1792: this lane's A3/A4).  PIX = pixel-changing, judged by the LPIPS gate in the TIER 3 header.",
    "HARNESS NOTE: every row runs the same lossless harness, which md5s the sbs (and anaglyph) arrays and writes FFV1",
    "  (~2.2 s of md5 at 576, ~3x that at 1792); the deployed CLI instead writes mp4v sbs+anaglyph (write_bench_v1: ~3.0 s at",
    "  576).  Ratios are like-for-like; absolute seconds carry the harness.",
]
lines.append("")
lines += open("scripts/distill/runs/more_20261004/pipeline_speed/TABLE_LIKE_FOR_LIKE.txt").read().rstrip().splitlines()
lines.append("P4 rep 1 (compiled decoder, 2nd clip) re-traced the new VAE object (window-0 decode 12.8 s) -> it is NOT a true")
lines.append("persistent-worker number; a worker keeping one compiled module is an ESTIMATE only (see the answer).")
out = sys.argv[1]
assert not os.path.exists(out), f"refusing to overwrite {out}"
open(out, "w").write("\n".join(lines) + "\n")
json.dump(J, open(out.replace(".txt", ".json"), "w"), indent=1)
print("\n".join(lines))
