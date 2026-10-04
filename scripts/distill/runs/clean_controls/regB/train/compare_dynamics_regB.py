"""Training-dynamics comparison of the three same-seed P1-style runs (identical sigma/window draw sequence):
   null   = minift/null   (target: origin's own mp4v output)
   pos    = minift/pos    (target: UNregistered real GT)
   regpos = regB/train/<run> (target: REGISTERED real GT)
Reports step-1 gradient norm (pre-clip), per-sigma mean loss / grad norm, and the fraction of steps that were clipped (gn>1).
usage: python compare_dynamics_regB.py <regpos run dir>"""
import sys, csv, collections
M = "/home/kawa/master_project/StereoCrafter/scripts/distill/runs/diag_trainer/minift"
runs = {"null": f"{M}/null/train_log.csv", "pos": f"{M}/pos/train_log.csv", "regpos": sys.argv[1].rstrip("/") + "/train_log.csv"}
rows = {k: list(csv.DictReader(open(p))) for k, p in runs.items()}
# the draw sequence must be identical (same seed): check sigma/window per step
for k in ("pos", "regpos"):
    same = all(a["sigma"] == b["sigma"] and a["win_start"] == b["win_start"] for a, b in zip(rows["null"], rows[k]))
    print(f"draw sequence identical to null: {k}: {same}")
print(f"\n{'run':8s} {'step1 gn':>9s} {'step1 loss':>10s} {'mean loss':>10s} {'mean gn':>8s} {'clipped%':>9s} {'last50 loss':>11s}")
for k, r in rows.items():
    gn = [float(x["grad_norm_preclip"]) for x in r]; ls = [float(x["loss"]) for x in r]
    print(f"{k:8s} {gn[0]:9.3f} {ls[0]:10.4f} {sum(ls)/len(ls):10.4f} {sum(gn)/len(gn):8.3f} {100*sum(g>1 for g in gn)/len(gn):8.1f}% {sum(ls[-50:])/50:11.4f}")
print("\nper-sigma mean loss (null / pos / regpos) and mean pre-clip grad norm:")
sig = sorted({x["sigma"] for x in rows["null"]}, key=float, reverse=True)
print(f"{'sigma':>8s} " + " ".join(f"{k+'_loss':>12s}" for k in rows) + " | " + " ".join(f"{k+'_gn':>10s}" for k in rows))
for s in sig:
    cells = []; gcells = []
    for k, r in rows.items():
        sel = [x for x in r if x["sigma"] == s]
        cells.append(sum(float(x["loss"]) for x in sel) / len(sel)); gcells.append(sum(float(x["grad_norm_preclip"]) for x in sel) / len(sel))
    print(f"{float(s):8.4g} " + " ".join(f"{c:12.4f}" for c in cells) + " | " + " ".join(f"{g:10.3f}" for g in gcells) + f"   (n={len(sel)})")
