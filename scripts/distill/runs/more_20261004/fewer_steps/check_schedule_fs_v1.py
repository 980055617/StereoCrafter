#!/usr/bin/env python
"""C4 schedule check: every render of this lane used exactly the schedule its label names.
For label <model>_g100_<SCHED>pad[_...] the expected sigmas are sched_defs.txt's list + [0.0]; checks: logged sigmas (exact
reprs), default8_index, UNet calls per window (N for plain renders; oracle renders are reported, not asserted), batch [1]
at guidance 1.00, pad draws per window == 8 - N.
usage: check_schedule_fs_v1.py OUT.txt
"""
import glob, json, os, re, sys
os.chdir("/home/kawa/master_project/StereoCrafter")
OUT = sys.argv[1]
assert not os.path.exists(OUT), f"refusing to overwrite {OUT}"
IDX = {"700.0": 0, "30.993608474731445": 3, "7.276163101196289": 4, "1.1675708293914795": 5, "0.09738767892122269": 6,
       "0.0020000000949949026": 7}     # exact reprs of default-8 entries (indices from the speed lane's logged default8_index)
defs = {}
for line in open("scripts/distill/runs/more_20261004/fewer_steps/sched_defs.txt"):
    if line.strip():
        k, v = line.split()
        defs[k] = v.split(",")
L, nfail = [], 0
for sp in sorted(glob.glob("outputs/more_20261004/fewer_steps/clips/*/speed_log.json")):
    d = os.path.basename(os.path.dirname(sp))
    j = json.load(open(sp))
    m = re.search(r"_g10[01]_(T\d[a-d]?)pad", d)
    if not m:
        L.append(f"SKIP {d}: label names no schedule")
        continue
    sched = m.group(1)
    exp = defs[sched]
    n = len(exp)
    sc = j["schedule"] or {}
    got = None
    for dev, rec in sc.items():
        if dev.startswith("cuda"):
            got = rec
    ok_sig = got is not None and got["sigmas"] == exp + ["0.0"]
    ok_idx = got is not None and got["default8_index"] == [IDX[s] for s in exp]
    calls = j["unet_calls_per_window"]
    is_oracle = os.path.exists(os.path.join(os.path.dirname(sp), "oracle_diag.json"))
    ok_calls = is_oracle or calls == [n]
    ok_batch = j["batch_sizes"] == [1] if abs(j["guid"] - 1.0) < 1e-12 else True
    pads = sorted(set(w.get("pad_draws", 0) for w in j["windows"]))
    ok_pad = pads == [8 - n]
    ok = ok_sig and ok_idx and ok_calls and ok_batch and ok_pad
    nfail += (not ok)
    L.append(f"{'PASS' if ok else 'FAIL'} {d:34s} sched={sched} N={n} sigmas_ok={ok_sig} idx={got['default8_index'] if got else None} "
             f"calls/window={calls}{' (oracle: not asserted)' if is_oracle else ''} batch={j['batch_sizes']} "
             f"pad/window={pads} windows={j['n_windows']} unet_s={j['unet_ms_sum']/1000:.2f} call_s={j['call_s_sum']:.2f} "
             f"hook_total_s={j['hook_total_s']:.1f}")
L.append(f"SUMMARY checked={sum(1 for x in L if x[:4] in ('PASS', 'FAIL'))} failed={nfail}")
open(OUT, "w").write("\n".join(L) + "\n")
print("\n".join(L))
