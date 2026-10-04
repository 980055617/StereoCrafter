#!/usr/bin/env python
"""Warm-up diagnosis table (F5) from the B1 runs: first UNet forward, autotune lines, latent identity.
usage: table_warmup_v1.py <out_txt (new)>"""
import json, os, re, sys
O = "outputs/more_20261004/pipeline_speed"
RUNS = [("s0  default ~/.triton/cache (warm), 2 windows", f"{O}/s0_smoke/clips/0301_deliv_T5_smoke2win", 0),
        ("B1b EMPTY TRITON_CACHE_DIR (cold compile)", f"{O}/b1_warmup/clips/0301_B1b_coldcache", 0),
        ("B1c same TRITON_CACHE_DIR again (warm)", f"{O}/b1_warmup/clips/0301_B1c_samecache_warm", 0),
        ("B1d autotune choices pre-filled (SK_AT_LOAD)", f"{O}/b1_warmup/clips/0301_B1d_atload", 0),
        ("B1e two clips in ONE process: clip 1", f"{O}/b1_warmup/clips/0301_B1e_repeat2", 0),
        ("B1e two clips in ONE process: clip 2", f"{O}/b1_warmup/clips/0301_B1e_repeat2", 1),
        ("A1  full clip, default cache (deployed path)", f"{O}/a_576/clips/0301_A1_deliv_T5_576", 0),
        ("C1a full clip, identity set incl. SK_AT_LOAD", f"{O}/c_576/clips/0301_C1a_deliv_T5_576_idfix", 0)]
out = sys.argv[1]
assert not os.path.exists(out)
L = [f"  {'run':48s} {'1st UNet fwd ms':>15s} {'2nd fwd ms':>10s} {'autotune s (printed)':>20s} {'md5':>9s}"]
for label, d, rep in RUNS:
    S = json.load(open(os.path.join(d, "stage_log.json")))
    r = [x for x in S["runs"] if x["rep"] == rep][0]
    e = r["windows"][0]["unet_ms_each"]
    log = open(d + ".log").read() if os.path.exists(d + ".log") else ""
    at = [float(x) for x in re.findall(r"Triton autotuning for function \S+ finished after ([0-9.]+)s", log)]
    if label.startswith("B1e") and rep == 1:
        at_s = "none (0 new lines)"
    else:
        at_s = (f"{sum(at):.2f} ({len(at)} kernels)" if at else ("not printed" if "TRITON_PRINT" not in json.dumps(S["env"]) or S["env"].get("TRITON_PRINT_AUTOTUNING") != "1" else "none"))
    L.append(f"  {label:48s} {e[0]:15.0f} {e[1]:10.0f} {at_s:>20s} {r['writer_md5'][0].split()[0][:8]:>9s}")
L.append("")
L.append("s0/B1b/B1c/B1d/B1e(both clips): saved final latents of windows 0 and 1 are bit-identical (compare_latents_v1.py),")
L.append("2-window sbs md5 719d6ceb in all of them; A1 (14 windows) chose a different _state_passing_fwd_kernel block size")
L.append("(1024 vs 512) than s0 and its latents are still bit-identical -> the remaining autotune choice does not change bits.")
open(out, "w").write("\n".join(L) + "\n")
print("\n".join(L))
