#!/usr/bin/env python
"""I3: compare per-window initial-latent fingerprints (speed_log.json init_md5) of render dirs against a reference
render dir (same clip).  A truncated (smoke) render is compared on its own windows only.
usage: check_fingerprints_v1.py OUT.txt REF_DIR DIR [DIR ...]"""
import json, os, sys
out, ref, dirs = sys.argv[1], sys.argv[2], sys.argv[3:]
R = json.load(open(os.path.join(ref, "speed_log.json")))
lines, nfail = [], 0
for d in dirs:
    J = json.load(open(os.path.join(d, "speed_log.json")))
    a, b = J["init_md5"], R["init_md5"]
    n = len(a)
    same = [a[i] == b[i] for i in range(min(n, len(b)))]
    ok = all(same) and n <= len(b) and (J["maxchunks"] is not None or n == len(b))
    nfail += (not ok)
    lines.append(f"{'PASS' if ok else 'FAIL'} {d} vs {ref}: windows {n} (ref {len(b)}), identical {sum(same)}/{len(same)}"
                 f"  sampler={J['sampler']} N={J['N']} pad={J['pad_to']} draws/window={J['draws_per_window']}"
                 f" unet_calls/window={J['unet_calls_per_window']} batch={J['batch_sizes']}"
                 + ("" if all(same) else f"  first mismatch at window {same.index(False)}"))
with open(out, "a") as fh:
    for ln in lines:
        fh.write(ln + "\n"); print(ln)
    fh.write(f"SUMMARY checked={len(dirs)} failed={nfail}\n")
print(f"SUMMARY checked={len(dirs)} failed={nfail}")
