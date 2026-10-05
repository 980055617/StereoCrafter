#!/usr/bin/env python
"""Gate for the scale_gt registration estimator (prep_crops_v1.py --reg-only, 8 frames, train-tile BR, coarse-to-fine):
compare its clip-global shift with (a) regB anchors 0301 (0,-12) / 0204 (-1,-17) and (b) eval_robustness REG_CLIP optima
(outputs/more_20261004/eval_robustness/score_v1/<clip>.json reg.clip_ddy/clip_ddx; all frames, splatting BR, exhaustive).
PASS rule (set before looking at (b)): |d ddy| <= 1 and |d ddx| <= 2 on >= 11/12 clips, and never > 4 px."""
import json, os, sys
os.chdir("/home/kawa/master_project/StereoCrafter")
G = "/mnt/ssd_data/deep_20261004/scale_gt/regGate_v2/reg_only"
rows = []; ok = 0; worst = 0
for c in ["0042", "0052", "0125", "0128", "0141", "0147", "0170", "0204", "0225", "0251", "0259", "0301"]:
    m = json.load(open(f"{G}/{c}/clip.json"))
    e = json.load(open(f"outputs/more_20261004/eval_robustness/score_v1/{c}.json"))["reg"]
    dy, dx = m["ddy"] - e["clip_ddy"], m["ddx"] - e["clip_ddx"]
    good = abs(dy) <= 1 and abs(dx) <= 2; ok += good; worst = max(worst, abs(dx), abs(dy))
    rows.append(f"{c}  mine ({m['ddy']:+d},{m['ddx']:+d}) psnr {m['mean_psnr']:.2f} (zero {m['mean_psnr_zero']:.2f})  "
                f"eval_robustness REG_CLIP ({e['clip_ddy']:+d},{e['clip_ddx']:+d}) psnr {e['clip_mean_psnr']:.2f}  diff ({dy:+d},{dx:+d}) {'ok' if good else 'MISS'}")
print("\n".join(rows))
anch = {"0301": (0, -12), "0204": (-1, -17)}
for c, (ay, ax) in anch.items():
    m = json.load(open(f"{G}/{c}/clip.json")); print(f"regB anchor {c}: mine ({m['ddy']:+d},{m['ddx']:+d}) vs regB ({ay:+d},{ax:+d})")
print(f"GATE {'PASS' if ok >= 11 and worst <= 4 else 'FAIL'}: {ok}/12 within (1,2) px, worst |diff| {worst} px")
