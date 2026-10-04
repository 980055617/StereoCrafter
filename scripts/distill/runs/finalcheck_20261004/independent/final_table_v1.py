#!/usr/bin/env python
"""finalcheck_20261004 / independent lane -- the final thesis table, built ONLY from this lane's own files:
  quality  : ANALYSIS_R1.json (this lane's score_clip_ll.py re-score, 12 clips, 576x1024)
  timing   : TIMING_H3.json (576x1024 recomputed from the speed lane's speed_log.json; hi-res from this lane's renders)
usage: final_table_v1.py OUT.txt ANALYSIS_R1.json TIMING_H3.json
"""
import json, os, statistics as st, sys

os.chdir("/home/kawa/master_project/StereoCrafter")
outp, ar1, tim = sys.argv[1], sys.argv[2], sys.argv[3]
assert not os.path.exists(outp), "refusing to overwrite"
A = json.load(open(ar1))["summary"]
T = json.load(open(tim))
r576 = {k: v[0] for k, v in T["res576"].items()}           # steady s/window, median over renders
hr = {r["dir"]: r for r in T["rows"]}


def rel576(lab):
    return r576[lab] / r576["origin_g101_s8"] if lab in r576 else None


def relhr(dirlab, res):
    ref = hr.get(f"0204_origin_g101_s8_{res}")
    r = hr.get(f"0204_{dirlab}_{res}")
    return (r["steady_ms"] / ref["steady_ms"]) if (r and ref) else None


ROWS = [("deployed origin (8 steps, guidance 1.01)", "origin_ll", "origin_g101_s8", "origin_g101_s8"),
        ("origin, recommended settings (T5, guidance 1.00)", "origin_g100_T5pad", "origin_g100_T5pad", "origin_g100_T5pad"),
        ("shipped 5-slot Mamba (old), deployed settings", "mamba_ll", "deliv_g101_s8", "deliv_g101_s8"),
        ("DELIVERABLE, deployed settings", "mstudent2_step800_deliv_ll", "deliv_g101_s8", "deliv_g101_s8"),
        ("DELIVERABLE, recommended settings (T5, g 1.00, RNG-paired)", "deliv_g100_T5pad", "deliv_g100_T5pad", "deliv_g100_T5pad"),
        ("  same, unpadded RNG (the literal shipped sampler)", "deliv_g100_T5nat", "deliv_g100_T5pad", "deliv_g100_T5pad")]
L = []; P = L.append
P("FINAL TABLE -- 12 test clips, lossless FFV1, real-GT LPIPS (score_clip_ll.py SCORE_STEP=4), all numbers from this lane")
P(f"{'row':60s} {'LPIPS12':>8s} {'d vs dep.origin':>15s} {'improved':>9s} {'sharp/GT':>9s} {'UNet/win 576':>13s} {'1024x1792':>10s} {'1024x1920':>10s}")
for name, qlab, t576, thr in ROWS:
    a = A[qlab]
    u = rel576(t576); u1 = relhr(thr, "1024x1792"); u2 = relhr(thr, "1024x1920")
    f = lambda x: f"{x:.3f}" if x is not None else "n/a"
    P(f"{name:60s} {a['mean']:8.4f} {a['d_origin']:+15.4f} {a['better']:6d}/12 {a['sh_gt']:9.3f} {f(u):>13s} {f(u1):>10s} {f(u2):>10s}")
P("UNet/win = CUDA-event UNet time per 14-frame window, steady state (median of windows 1..n-2), relative to deployed origin "
  "measured the same way at the same resolution; 576x1024 from the speed lane's renders (median over its renders), hi-res "
  "from this lane's clip-0204 renders.  The shipped Mamba row reuses the deliverable 8x2 timing (same 5-slot architecture).")
open(outp, "w").write("\n".join(L) + "\n")
print("\n".join(L))
