#!/usr/bin/env python
"""vae_20261005 / decoder_swap -- GATE D1 / T-D1 (PREREG.txt sections 3, 4): this lane's stock re-decode SBS md5 == the
decoder_ft capture render md5 (C1-gated by decoder_ft against the published lossless renders) for every listed cell.
Also lists, for information, the md5 of every other row of each cell (they must DIFFER from stock unless identical decoders).
usage: python check_md5_swap_v1.py <out_txt> <redec_root> <clip,...>
"""
import os, sys
CAP = "/mnt/ssd_data/deep_20261004/decoder_ft/capture/clips"
OUT, RED, CLIPS = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
assert not os.path.exists(OUT), OUT
def md5_of(d):
    p = [f for f in os.listdir(d) if f.endswith("_sbs.mkv.md5")] if os.path.isdir(d) else []
    return open(os.path.join(d, p[0])).read().split()[0] if p else None
lines, ok = [], True
for c in CLIPS:
    for lat in ("origin_cap", "deliv_cap"):
        cm = md5_of(f"{CAP}/{c}_{lat}")
        st = md5_of(f"{RED}/{c}_{lat}__stock")
        g = (cm is not None) and st == cm
        ok &= g
        others = sorted(d.split("__")[1] for d in os.listdir(RED) if d.startswith(f"{c}_{lat}__") and not d.endswith("__stock"))
        om = {o: md5_of(f"{RED}/{c}_{lat}__{o}") for o in others}
        lines.append(f"{c} {lat:10s} capture {cm} stock re-decode {st} -> {'PASS' if g else 'FAIL'}; other rows "
                     + ", ".join(f"{o} {'differs' if m != cm else 'IDENTICAL'}" for o, m in om.items()))
lines.append(f"GATE D1 {'ALL PASS' if ok else 'FAILURE'}")
open(OUT, "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
