"""Turn ROW lines from score_clip_ll.py into the Part-1 / Part-2 tables."""
import re, sys, statistics as st

rows = {}   # (clip, suffix) -> dict
for path in sys.argv[1:]:
    for line in open(path):
        if not line.startswith("ROW "):
            continue
        d = dict(kv.split("=", 1) for kv in line.split()[1:] if "=" in kv)
        clip = d["clip"]; suf = d["tag"][len(clip) + 1:]
        rows[(clip, suf)] = {k: d[k] for k in d}

CLIPS12 = "0042 0052 0125 0128 0141 0147 0170 0204 0225 0251 0259 0301".split()
FOUR = ["0301", "0204", "0052", "0147"]
F = lambda c, s, k: float(rows[(c, s)][k])
have = lambda c, s: (c, s) in rows

# mp4v reference deltas, from scripts/distill/runs/fulldata_v2/beyond3/s25_12clip_summary.txt
MP4V = {  # clip: (origin_lpips, s25_lpips, g_none)
 "0042": (0.4337, 0.4182), "0052": (0.4432, 0.4368), "0125": (0.4768, 0.4506),
 "0128": (0.4146, 0.3928), "0141": (0.4709, 0.4612), "0147": (0.5183, 0.5176),
 "0170": (0.2617, 0.2531), "0204": (0.2122, 0.1950), "0225": (0.2837, 0.2681),
 "0251": (0.3505, 0.3418), "0259": (0.4397, 0.4406), "0301": (0.4445, 0.4083)}

def table(title, base, var, clips):
    clips = [c for c in clips if have(c, base) and have(c, var)]
    if not clips: return None
    print(f"\n=== {title}  ({var} vs {base}, LOSSLESS FFV1, real-GT LPIPS, SCORE_STEP=4) ===")
    print(f"{'clip':5s} {'base':>8s} {'var':>8s} {'delta':>8s} | {'sh_base':>8s} {'sh_var':>8s} {'sh_GT':>8s} "
          f"{'var/GT':>7s} {'overshoot?':>11s} | {'lPSNRb':>8s} {'lPSNRv':>8s} {'rPSNRb':>8s} {'rPSNRv':>8s}")
    ds, ratios, over = [], [], []
    for c in clips:
        b, v = F(c, base, "lpips"), F(c, var, "lpips")
        sb, sv, sg = F(c, base, "sharp"), F(c, var, "sharp"), F(c, base, "gtSharp")
        d = v - b; ds.append(d); ratios.append(sv / sb)
        os_ = sv > sg; over.append(os_)
        print(f"{c:5s} {b:8.4f} {v:8.4f} {d:+8.4f} | {sb:8.4f} {sv:8.4f} {sg:8.4f} {sv/sg:7.3f} "
              f"{('OVERSHOOT' if os_ else 'under-GT'):>11s} | {F(c,base,'leftPSNR'):8.2f} {F(c,var,'leftPSNR'):8.2f} "
              f"{F(c,base,'rightPSNR'):8.3f} {F(c,var,'rightPSNR'):8.3f}")
    mb = st.fmean(F(c, base, "lpips") for c in clips); mv = st.fmean(F(c, var, "lpips") for c in clips)
    print(f"{'MEAN':5s} {mb:8.4f} {mv:8.4f} {mv-mb:+8.4f}   n={len(clips)}  improved={sum(1 for d in ds if d<0)}/{len(ds)}"
          f"  worst={max(ds):+.4f}  best={min(ds):+.4f}  meanSharpRatio={st.fmean(ratios):.3f}"
          f"  overshootGT={sum(over)}/{len(over)}")
    bad = [c for c in clips if abs(F(c, base, "leftPSNR") - F(c, var, "leftPSNR")) > 1e-4]
    print(f"leftPSNR identical base-vs-var on all {len(clips)} clips: {not bad}" + (f"  DIFFER: {bad}" if bad else ""))
    offs = [c for c in clips if (rows[(c,base)]['dy'],rows[(c,base)]['dx']) != (rows[(c,var)]['dy'],rows[(c,var)]['dx'])]
    print(f"alignment offset identical base-vs-var: {not offs}" + (f"  DIFFER: {offs}" if offs else ""))
    return dict(clips=clips, ds=ds, mb=mb, mv=mv)

r1 = table("PART 1  s25 on the four regime-spanning clips", "origin_ll", "s25_ll", FOUR)
if r1:
    print("\n--- PART 1  lossless delta vs mp4v delta, side by side ---")
    print(f"{'clip':5s} {'LL origin':>10s} {'LL s25':>9s} {'LL delta':>9s} | {'mp4v origin':>12s} {'mp4v s25':>9s} {'mp4v delta':>11s} | {'delta of deltas':>15s}")
    for c in r1["clips"]:
        lb, lv = F(c, "origin_ll", "lpips"), F(c, "s25_ll", "lpips")
        mo, ms = MP4V[c]
        print(f"{c:5s} {lb:10.4f} {lv:9.4f} {lv-lb:+9.4f} | {mo:12.4f} {ms:9.4f} {ms-mo:+11.4f} | {(lv-lb)-(ms-mo):+15.4f}")
    print(f"{'MEAN':5s} {st.fmean(F(c,'origin_ll','lpips') for c in r1['clips']):10.4f} "
          f"{st.fmean(F(c,'s25_ll','lpips') for c in r1['clips']):9.4f} "
          f"{st.fmean(F(c,'s25_ll','lpips')-F(c,'origin_ll','lpips') for c in r1['clips']):+9.4f} | "
          f"{st.fmean(MP4V[c][0] for c in r1['clips']):12.4f} {st.fmean(MP4V[c][1] for c in r1['clips']):9.4f} "
          f"{st.fmean(MP4V[c][1]-MP4V[c][0] for c in r1['clips']):+11.4f}")

r2 = table("PART 2  g125 on all 12 test clips", "origin_ll", "g125_ll", CLIPS12)
if r2:
    print("\n--- PART 2  g125 lossless delta vs the mp4v 3-clip smoke (beyond2/SUMMARY_TABLE.txt) ---")
    SMOKE = {"0042": -0.0101, "0204": -0.0130, "0301": -0.0305}
    for c, m in SMOKE.items():
        if have(c, "g125_ll"):
            print(f"  {c}: lossless {F(c,'g125_ll','lpips')-F(c,'origin_ll','lpips'):+.4f}   mp4v smoke {m:+.4f}")
for v in ("g115_ll", "g140_ll"):
    table(f"PART 2  guidance optimum pinning: {v}", "origin_ll", v, FOUR)

print("\n--- guidance ladder on the four clips (lossless LPIPS) ---")
lad = ["origin_ll", "g115_ll", "g125_ll", "g140_ll", "s25_ll"]
print(f"{'clip':5s} " + " ".join(f"{x.replace('_ll',''):>9s}" for x in lad))
for c in FOUR:
    print(f"{c:5s} " + " ".join((f"{F(c,x,'lpips'):9.4f}" if have(c, x) else f"{'-':>9s}") for x in lad))
print(f"{'MEAN':5s} " + " ".join(
    (f"{st.fmean(F(c,x,'lpips') for c in FOUR):9.4f}" if all(have(c,x) for c in FOUR) else f"{'-':>9s}") for x in lad))
print(f"{'sharp':5s} " + " ".join(
    (f"{st.fmean(F(c,x,'sharp') for c in FOUR):9.4f}" if all(have(c,x) for c in FOUR) else f"{'-':>9s}") for x in lad))
