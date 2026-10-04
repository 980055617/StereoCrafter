#!/usr/bin/env python
"""Timing readout for this lane, SAME METHOD as finalcheck_20261004/speed/timing_table_v1.py's columns:
unet/win, steady/win = median over windows 1..n-2 of one render (drops window 0's warm-up and the short last window),
then median over renders; process_s = hook_total_s.  Every render here ran with GPU 0 held under flock for the whole
render (nothing else on GPU 0); GPU 1 was busy with other lanes (recorded in each <dir>.lockwait).
usage: timing_fs_v1.py OUT.txt
"""
import glob, json, os, statistics as st, sys
from collections import defaultdict
os.chdir("/home/kawa/master_project/StereoCrafter")
OUT = sys.argv[1]
assert not os.path.exists(OUT), f"refusing to overwrite {OUT}"
def rd(sp):
    j = json.load(open(sp)); W = j["windows"]; mid = W[1:-1]
    return dict(unet_win=st.median(w["unet_ms"] / 1000 for w in mid), steady_win=st.median(w["call_s"] for w in mid),
                unet_s=j["unet_ms_sum"] / 1000, inloop_s=j["call_s_sum"], process_s=j["hook_total_s"],
                calls=j["unet_calls_per_window"], batch=j["batch_sizes"], n=j["n_windows"])
L = []
P = L.append
rows = defaultdict(list)
for sp in sorted(glob.glob("outputs/more_20261004/fewer_steps/clips/*/speed_log.json")):
    d = os.path.basename(os.path.dirname(sp)); clip, lab = d.split("_", 1)
    rows["fewer_steps:" + lab].append((clip, rd(sp)))
for lab in ("deliv_g100_T5pad", "origin_g100_T5pad", "origin_g101_s8", "deliv_g101_s8"):
    for sp in sorted(glob.glob(f"outputs/finalcheck_20261004/speed/clips/*_{lab}/speed_log.json")):
        clip = os.path.basename(os.path.dirname(sp)).split("_", 1)[0]
        rows["speed:" + lab].append((clip, rd(sp)))
P(f"{'config':44s} {'n':>3s} {'calls/win':>9s} {'batch':>6s} {'unet/win':>9s} {'steady/win':>10s} {'process_s':>9s}  clips")
med = {}
for lab, rs in sorted(rows.items()):
    u = st.median(r["unet_win"] for _, r in rs); s = st.median(r["steady_win"] for _, r in rs)
    p = st.median(r["process_s"] for _, r in rs)
    med[lab] = (u, s, p)
    P(f"{lab:44s} {len(rs):3d} {str(rs[0][1]['calls']):>9s} {str(rs[0][1]['batch']):>6s} {u:9.3f} {s:10.3f} {p:9.1f}  "
      + " ".join(c for c, _ in rs))
P("")
P("PAIRED same-session (this lane), clip 0301: T4b vs the two T5 @1.00 pad renders made in this lane")
t4 = dict(rows["fewer_steps:deliv_g100_T4bpad"])["0301"]
for lab in ("deliv_g100_T5pad_rep", "deliv_g100_T5pad_orcC2nosubst"):
    t5 = dict(rows["fewer_steps:" + lab])["0301"]
    P(f"  T4b / {lab:32s} unet/win {t4['unet_win']:.3f}/{t5['unet_win']:.3f} = {t4['unet_win']/t5['unet_win']:.3f}   "
      f"steady/win {t4['steady_win']:.3f}/{t5['steady_win']:.3f} = {t4['steady_win']/t5['steady_win']:.3f}   "
      f"process {t4['process_s']:.1f}/{t5['process_s']:.1f} = {t4['process_s']/t5['process_s']:.3f}")
P("")
P("CROSS-SESSION (indicative only): medians of this lane's 12 T4b renders vs the speed lane's 12-clip medians")
u4, s4, p4 = med["fewer_steps:deliv_g100_T4bpad"]
for lab in ("speed:deliv_g100_T5pad", "speed:origin_g101_s8", "speed:deliv_g101_s8"):
    u, s, p = med[lab]
    P(f"  T4b / {lab:28s} unet/win {u4:.3f}/{u:.3f} = {u4/u:.3f}   steady/win {s4:.3f}/{s:.3f} = {s4/s:.3f}   process {p4:.1f}/{p:.1f} = {p4/p:.3f}")
open(OUT, "w").write("\n".join(L) + "\n")
print("\n".join(L))
