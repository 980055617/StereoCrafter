#!/usr/bin/env python
"""Headline end-to-end table (pre-registered H rule): process wall-clock of each config, speedup vs deployed origin,
md5 gate.  usage: table_headline_v1.py <out_txt (new)>   (paths are fixed below; missing renders are shown as '-')"""
import io, contextlib, json, os, statistics, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_stages_v2 as A

O = "outputs/more_20261004/pipeline_speed"
REF_MD5 = {"576_deliv": "a41b432baeb25393bb12d39b4f2e6a21", "576_origin": "2e533d7755c950d2fc95043f6fb0a51d"}
ROWS = {
    "576": [
        ("deployed origin 8x2@1.01", "origin", [f"{O}/a_576/clips/0301_A2_origin_s8_576", f"{O}/c_576/clips/0301_A2r_origin_s8_576"], 0),
        ("origin + identity fixes", "origin", [f"{O}/c_576/clips/0301_C3_origin_s8_576_idfix"], 0),
        ("deliverable T5@1.00", "deliv", [f"{O}/a_576/clips/0301_A1_deliv_T5_576", f"{O}/c_576/clips/0301_A1r_deliv_T5_576"], 0),
        ("deliverable T5@1.00 + identity fixes", "deliv", [f"{O}/c_576/clips/0301_C1a_deliv_T5_576_idfix",
                                                            f"{O}/c_576/clips/0301_C1b_deliv_T5_576_idfix_rep"], 0),
        ("  same, 2nd clip in a persistent process", "deliv", [f"{O}/c_576/clips/0301_C1w_deliv_T5_576_idfix_worker"], 1),
    ],
    "1792": [
        ("deployed origin 8x2@1.01", "origin", [f"{O}/a_1792/clips/0301_A4_origin_s8_1792"], 0),
        ("deliverable T5@1.00", "deliv", [f"{O}/a_1792/clips/0301_A3_deliv_T5_1792"], 0),
        ("deliverable T5@1.00 + identity fixes", "deliv", [f"{O}/c_1792/clips/0301_C2_deliv_T5_1792_idfix"], 0),
    ],
}


def load(d, rep):
    if not os.path.exists(os.path.join(d, "stage_log.json")):
        return None
    with contextlib.redirect_stdout(io.StringIO()):
        res = A.main([d])[d]
    r = [x for x in res["runs"] if x["rep"] == rep][0]
    return dict(ref=r["ref"], md5=r["md5"], ex=r["exclusive"], groups=r["groups"], w=r["windows"])


lines, J = [], {}
for res_name, rows in ROWS.items():
    lines.append(f"===== 0301 @ {'576x1024' if res_name == '576' else '1024x1792'} " + "=" * 80)
    lines.append(f"  {'config':44s} {'n':>2s} {'end-to-end s':>13s} {'(each)':>17s} {'speedup vs origin':>18s} "
                 f"{'UNet s':>8s} {'VAE dec s':>9s} {'md5 gate':>14s}")
    base = None
    ref_md5 = {}
    for label, model, dirs, rep in rows:
        got = [load(d, rep) for d in dirs]
        got = [g for g in got if g]
        if not got:
            lines.append(f"  {label:44s}  -  (not rendered)")
            continue
        m = statistics.mean(g["ref"] for g in got)
        if base is None:
            base = m
        md5s = sorted({g["md5"] for g in got})
        key = f"{res_name}_{model}"
        if key not in REF_MD5 and len(md5s) == 1 and "fixes" not in label and "persistent" not in label:
            REF_MD5[key] = md5s[0]
        gate = "-" if key not in REF_MD5 else ("PASS" if md5s == [REF_MD5[key]] else "FAIL " + ",".join(x[:8] for x in md5s))
        each = "/".join(f"{g['ref']:.1f}" for g in got)
        lines.append(f"  {label:44s} {len(got):>2d} {m:13.2f} {each:>17s} {base / m:17.3f}x "
                     f"{statistics.mean(g['ex'].get('unet_gpu', 0) for g in got):8.2f} "
                     f"{statistics.mean(g['ex'].get('vae_decode', 0) for g in got):9.2f} {gate:>14s}")
        J[f"{res_name}|{label}"] = dict(dirs=dirs, rep=rep, end_to_end_each=[g["ref"] for g in got], mean=m,
                                        speedup_vs_origin=base / m, md5=md5s, gate=gate)
    lines.append("")
lines.append("end-to-end = python start -> exit of a single-clip CLI process (driver wall-clock, GPU-0 lock held for the whole")
lines.append("render) ; persistent-process row = that clip's own run time inside a process that already rendered one clip")
lines.append("(model load still included, no python start-up/import, no autotune).  md5 gate: pre-encode sbs md5 equals the")
lines.append("reference render of the same model+resolution (576: speed-lane references; 1792: this lane's A3/A4).")
out = sys.argv[1]
assert not os.path.exists(out), f"refusing to overwrite {out}"
open(out, "w").write("\n".join(lines) + "\n")
json.dump(J, open(out.replace(".txt", ".json"), "w"), indent=1)
print("\n".join(lines))
