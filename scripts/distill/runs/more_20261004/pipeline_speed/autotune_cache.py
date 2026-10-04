"""Persist Triton autotune CHOICES across processes (Triton 3.0.0 keeps them only in memory: Autotuner.cache = {}).

save(path): for every live triton.runtime.autotuner.Autotuner, write {kernel: [{key, idx, cfg}]} where idx is the index of
            the chosen Config in that Autotuner's own .configs list (the list is built deterministically from source).
load(path): pre-fill Autotuner.cache[key] = configs[idx] (asserting str(config) matches what was saved), so Autotuner.run
            finds the key and skips benchmarking; the kernel then runs with the SAME config the saving process chose.
No Triton or mamba_ssm file is modified.  Kernel binaries keep coming from the normal on-disk cache (~/.triton/cache).
"""
import gc
import json

from triton.runtime.autotuner import Autotuner


def _name(at):
    fn = at.base_fn
    return f"{getattr(fn, '__module__', '?')}.{getattr(fn, '__qualname__', getattr(fn, '__name__', '?'))}"


def autotuners():
    out = {}
    for o in gc.get_objects():
        if isinstance(o, Autotuner):
            out.setdefault(_name(o), []).append(o)
    return out


def snapshot():
    data = {}
    for name, ats in autotuners().items():
        for at in ats:
            for key, cfg in at.cache.items():
                data.setdefault(name, []).append(dict(key=list(key), idx=at.configs.index(cfg), cfg=str(cfg),
                                                      bench_time=getattr(at, "bench_time", None)))
    return data


def save(path):
    data = snapshot()
    with open(path, "w") as fh:
        json.dump(data, fh, indent=1, sort_keys=True)
    return data


def load(path):
    with open(path) as fh:
        data = json.load(fh)
    ats = autotuners()
    n = 0
    for name, entries in data.items():
        assert name in ats, f"autotuned kernel {name} not found in this process"
        assert len(ats[name]) == 1, f"{name}: {len(ats[name])} Autotuner objects, ambiguous"
        at = ats[name][0]
        for e in entries:
            cfg = at.configs[e["idx"]]
            assert str(cfg) == e["cfg"], (name, str(cfg), e["cfg"])
            at.cache[tuple(e["key"])] = cfg
            n += 1
    return n
