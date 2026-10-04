"""Build TABLE_skeptic.txt: the answer table (deployed origin / origin+s25 / step-distilled student / control A step300 /
control B best step) from the skeptic's OWN re-scored ROW lines only (scores_skeptic_{0301,0204}.txt), plus every re-scored row."""
import os
REPO = "/home/kawa/master_project/StereoCrafter"; os.chdir(REPO)
SK = "scripts/distill/runs/clean_controls/skeptic"
rows = {}
for c in ("0301", "0204"):
    for line in open(f"{SK}/scores_skeptic_{c}.txt"):
        if line.startswith("ROW "):
            d = dict(kv.split("=", 1) for kv in line.strip().split()[1:]); rows[(c, d["tag"])] = d
def g(c, tag, k): return float(rows[(c, tag)][k])
spec = [
    ("deployed origin (8 Euler steps, g1.01, origin weights)",            "0301_origin_ll",           "0204_origin_ll",           "8 steps, shipped weights (1.0x UNet cost)"),
    ("origin + s25 (same weights, 25 steps)",                              "0301_s25_ll",              "0204_s25_ll",              "25 steps, shipped weights (3.1x UNet cost)"),
    ("step-distilled attn student (origin UNet, 8 steps)",                 "0301_student_ll",          "0204_student_ll",          "8 steps, swapped attn tensors (1.0x)"),
    ("mamba + mstudent1_step600 deliverable (8 steps)",                    "0301_mstudent1_step600_ll","0204_mstudent1_step600_ll","8 steps, Mamba blocks + distilled student (<1.0x UNet time)"),
    ("control A step300: lossless self-target fine-tune (llnull)",         "0301_llnull_step300",      "0204_llnull_step300",      "8 steps, 15 fine-tuned attn1 tensors (1.0x)"),
    ("control B best (step200 by sampled LPIPS): registered-GT fine-tune", "regB_step200_0301",        "regB_step200_0204",        "8 steps, 15 fine-tuned attn1 tensors (1.0x)"),
]
out = []
out.append("SKEPTIC RE-SCORE -- every number below was produced by beyond4/score_clip_ll.py (SCORE_STEP=4, LPIPS-alex vs the unregistered real right eye,")
out.append("alignment on the pass-through left half) run by the skeptic on the FFV1 renders on disk; files: scores_skeptic_0301.txt, scores_skeptic_0204.txt")
out.append("")
out.append(f"{'variant':72s} {'0301 LPIPS':>10s} {'0301 sharp':>10s} {'0204 LPIPS':>10s} {'0204 sharp':>10s}  what it needs at inference")
for name, t1, t2, need in spec:
    out.append(f"{name:72s} {g('0301',t1,'lpips'):10.4f} {g('0301',t1,'sharp'):10.4f} {g('0204',t2,'lpips'):10.4f} {g('0204',t2,'sharp'):10.4f}  {need}")
out.append("")
out.append("deltas vs deployed origin (LPIPS):")
o1, o2 = g("0301", "0301_origin_ll", "lpips"), g("0204", "0204_origin_ll", "lpips")
for name, t1, t2, _ in spec[1:]:
    out.append(f"  {name:70s} 0301 {g('0301',t1,'lpips')-o1:+.4f}   0204 {g('0204',t2,'lpips')-o2:+.4f}")
out.append("")
out.append("ALL re-scored rows (tag, offset, n, LPIPS, sharp, rPSNR):")
for (c, tag), d in sorted(rows.items()):
    out.append(f"  {c} {tag:36s} ({d['dy']},{d['dx']}) n={d['n']}  lpips {float(d['lpips']):.4f}  sharp {float(d['sharp']):.4f}  rPSNR {float(d['rightPSNR']):.3f}")
open(f"{SK}/TABLE_skeptic.txt", "w").write("\n".join(out) + "\n")
print("\n".join(out))
