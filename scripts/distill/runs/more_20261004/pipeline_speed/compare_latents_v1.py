#!/usr/bin/env python
"""Bit-compare saved per-window final latents (<dir>/lat/wNN.pt) between a reference run and other runs.
usage: compare_latents_v1.py <ref_dir> <dir> [<dir> ...]"""
import hashlib, os, sys
import torch


def md5(t):
    t = t.contiguous()
    return hashlib.md5((t.view(torch.int16) if t.element_size() == 2 else t).numpy().tobytes()).hexdigest()


ref = sys.argv[1]
rl = sorted(os.listdir(os.path.join(ref, "lat")))
for d in sys.argv[2:]:
    for sub in [d] + sorted(os.path.join(d, x) for x in os.listdir(d) if x.startswith("rep") and os.path.isdir(os.path.join(d, x))):
        ld = os.path.join(sub, "lat")
        if not os.path.isdir(ld):
            continue
        fl = sorted(os.listdir(ld))
        res = []
        for f in fl:
            if f in rl:
                a, b = torch.load(os.path.join(ref, "lat", f)), torch.load(os.path.join(ld, f))
                res.append(f"{f}:{'EQUAL' if torch.equal(a, b) else 'DIFF(maxabs=%.3g)' % (a.float() - b.float()).abs().max().item()}")
        print(f"{sub} vs {ref}: {' '.join(res)}  (ref has {len(rl)}, this has {len(fl)})")
