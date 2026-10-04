"""Consolidated CONTROL B table from the machine-readable ROW / ROWREG lines of the score files in this directory.
Every number is read from a file; the source file is printed with each block.  usage: python summarize_regB.py"""
import os, re, glob
S = os.path.dirname(os.path.abspath(__file__))
def rows(path, key="ROW "):
    out = {}
    for ln in open(path):
        if ln.startswith(key):
            d = dict(kv.split("=", 1) for kv in ln.split()[1:] if "=" in kv)
            out[(d["clip"], d["tag"])] = d
    return out
std = {}; reg = {}
for f in sorted(glob.glob(f"{S}/scores_*.txt")) + sorted(glob.glob(f"{S}/early_*.txt")):
    if "regdiag" in f: reg.update(rows(f, "ROWREG "))
    else: std.update(rows(f, "ROW "))
ctx = {"0301": [("0301_origin_ll", "deployed origin"), ("0301_s25_ll", "origin+s25 (25 steps)"), ("0301_student_ll", "origin+attn student (step-distilled)"), ("0301_mstudent1_step600_ll", "mamba+mstudent1_step600 (deliverable)")],
       "0204": [("0204_origin_ll", "deployed origin"), ("0204_s25_ll", "origin+s25 (25 steps)"), ("0204_student_ll", "origin+attn student (step-distilled)"), ("0204_mstudent1_step600_ll", "mamba+mstudent1_step600 (deliverable)")]}
def fmt(clip, tag, label, base):
    r = std.get((clip, tag)); q = reg.get((clip, tag))
    if r is None: return f"  {label:46s} {'(no row)':>8s}"
    lp = float(r["lpips"]); d = lp - base
    s = f"  {label:46s} {lp:8.4f} {d:+8.4f} {float(r['sharp']):8.4f} {float(r['rightPSNR']):8.3f}"
    s += f" {float(q['lpips_reg']):9.4f} {float(q['rightPSNR_reg']):9.3f}" if q else f" {'':>9s} {'':>9s}"
    return s
for clip in ("0301", "0204"):
    base = float(std[(clip, f"{clip}_origin_ll")]["lpips"]); gts = float(std[(clip, f"{clip}_origin_ll")]["gtSharp"])
    print(f"\n=== {clip}  (GT sharp {gts:.4f}; LPIPS = standard score_clip_ll.py vs UNregistered GT, SCORE_STEP=4, n=38; LPIPSreg/rPSNRreg = diagnostic vs REGISTERED GT) ===")
    print(f"  {'row':46s} {'LPIPS':>8s} {'d vs org':>8s} {'sharp':>8s} {'rPSNR':>8s} {'LPIPSreg':>9s} {'rPSNRreg':>9s}")
    for tag, lab in ctx[clip]: print(fmt(clip, tag, lab, base))
    print("  -- CONTROL B: registered-GT training (regpos, trained on 0301 only) --")
    for tag in sorted({t for (c, t) in std if c == clip and t.startswith("regB_")}, key=lambda t: (len(t), t)): print(fmt(clip, tag, tag, base))
    pf = sorted({t for (c, t) in std if c == clip and t.startswith("regBpf_")}, key=lambda t: (len(t), t))
    if pf:
        print("  -- follow-up: PER-FRAME registered GT (regBpf) --")
        for tag in pf: print(fmt(clip, tag, tag, base))
    pu = sorted({t for (c, t) in std if c == clip and t.startswith("posUnreg_")})
    if pu:
        print("  -- reference: UNregistered-GT P1 run (minift/pos checkpoints), re-rendered losslessly --")
        for tag in pu: print(fmt(clip, tag, tag, base))
print("\nsource files:", ", ".join(os.path.basename(f) for f in sorted(glob.glob(f"{S}/scores_*.txt")) + sorted(glob.glob(f"{S}/early_*.txt"))))
