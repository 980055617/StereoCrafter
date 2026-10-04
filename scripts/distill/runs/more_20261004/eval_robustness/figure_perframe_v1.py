#!/usr/bin/env python
"""Per-frame figure: deliverable - origin LPIPS per scored frame, 12 clips as small multiples, UNREG vs REG_FRAME.
usage: python figure_perframe_v1.py <score_dir> <out.png>
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCORE, OUT = sys.argv[1], sys.argv[2]
WIDE = sys.argv[3] if len(sys.argv) > 3 else ""      # PREREG_ADDENDUM_boundary.txt override (widened clips)
assert not os.path.exists(OUT), f"refusing to overwrite {OUT}"
CLIPS = "0042 0052 0125 0128 0141 0147 0170 0204 0225 0251 0259 0301".split()
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#d9d8d4"
C_REG, C_UN = "#2a78d6", "#eb6834"          # categorical slots 1 and 2 of the reference palette
plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "text.color": INK, "figure.facecolor": SURF, "axes.facecolor": SURF})
fig, axs = plt.subplots(3, 4, figsize=(15, 8.8), sharey=False)
lo, hi = 0, 0
for ax, c in zip(axs.flat, CLIPS):
    pw = os.path.join(WIDE, f"{c}.json") if WIDE else ""
    d = json.load(open(pw if pw and os.path.exists(pw) else os.path.join(SCORE, f"{c}.json")))
    fr = np.asarray(d["frames"])
    cf = d["configs"]
    for var, col, lab in (("UNREG", C_UN, "unregistered GT (published metric)"),
                          ("REG_FRAME", C_REG, "registered GT (per-frame shift)")):
        y = np.asarray(cf["mstudent2_step800_deliv_ll"]["lpips"][var]) - np.asarray(cf["origin_ll"]["lpips"][var])
        lo, hi = min(lo, y.min(), 0), max(hi, y.max(), 0)
        ax.plot(fr, y, color=col, lw=1.6, marker="o", ms=3.2, mec=SURF, mew=0.8, label=lab, zorder=3)
        ax.axhline(y.mean(), color=col, lw=0.9, ls=(0, (4, 3)), zorder=2)
    # sampler window starts (14-frame windows, stride 11) as faint guides
    wk = np.asarray(d["window_k"])
    for k in range(1, wk.max() + 1):
        f0 = fr[wk == k].min() if np.any(wk == k) else None
        if f0 is not None:
            ax.axvline(f0 - 2, color=GRID, lw=0.6, zorder=1)
    ax.axhline(0, color=INK2, lw=0.8, zorder=2)
    ax.set_title(f"{c}   GT shift {d['reg']['clip_ddx']:+d} px (clip)", fontsize=9.5, color=INK, loc="left")
    ylo, yhi = ax.get_ylim()
    ax.set_ylim(min(ylo, 0), max(yhi, 0) + 0.05 * (yhi - ylo))
    lo, hi = 0, 0
    ax.grid(axis="y", color=GRID, lw=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
for ax in axs[-1]:
    ax.set_xlabel("frame index")
for ax in axs[:, 0]:
    ax.set_ylabel("LPIPS(deliverable) - LPIPS(origin)\n(below 0 = deliverable better)")
h, l = axs[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="upper left", bbox_to_anchor=(0.005, 0.958), ncol=2, frameon=False, fontsize=9.5)
fig.suptitle("Per-frame LPIPS change of the deliverable vs deployed origin, 12 test clips. Dashed = clip mean; faint "
             "verticals = sampler window starts; y-axes are independent per clip", x=0.01, y=0.995, ha="left",
             fontsize=11, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.925))
fig.savefig(OUT, dpi=130, facecolor=SURF)
print("wrote", OUT)
