"""Fixed clip split + nested scaling-curve subsets for full-data attention->Mamba distillation (fulldata_v1).

Reads scripts/distill/runs/clip_inventory.json, writes scripts/distill/splits/fulldata_v1.json.
Deterministic (SEED). CPU only. Re-running must reproduce the committed file byte-for-byte.
(Distinct from make_split.py / runs/heldout_v1.json, a sibling proposal with 152-frame segments.)

Rules
  excluded : 0312 (unreadable), 960x1280 clips (per-eye 480x640 < 576x1024 crop)
  test     : LPIPS + relMSE, never trained. gt=True, frames<=200, 6 per tile format,
             forced members 0042 (4400) / 0204 / 0301 (2160) = the clips that were never distilled on.
  dev      : relMSE only (model selection / stop rule), never trained. 3 short per format + 2 long (>400 f).
  train    : everything else (incl. 0160 = contaminated continuity reference, all long clips, 0310/0311).
  curve    : nested subsets 13 / 40 / 120 / all of the train pool, stratified by
             (format, long) with one seeded shuffle per stratum, largest-remainder allocation,
             0160 forced into the 13-point. Point k is a prefix of point k+1 and of train_order,
             so capture order == train_order gives every point its data as early as possible.
  windows  : per-clip capture budget W(frames) = 2 (<=200) / 3 (<=400) / 4 (<=1000) / 6 (>1000),
             chosen evenly spaced on the DEPLOYED window grid (frames_chunk 14, overlap 3, stride 11,
             last window pulled back to end at n-1 exactly like inpainting_inference.main).
"""
import json, os, random, hashlib

SEED = 20260918
HERE = os.path.dirname(os.path.abspath(__file__))
INV = os.path.join(HERE, "runs", "clip_inventory.json")
OUT = os.path.join(HERE, "splits", "fulldata_v1.json")
FORCED_TEST = {"0042", "0204", "0301"}
CONTINUITY = "0160"
N_TEST_PER_FMT = 6
N_DEV_SHORT_PER_FMT = 3
N_DEV_LONG = 2
CURVE_POINTS = [13, 40, 120]  # + "all"
SHORT_MAX = 200
LONG_MIN = 401  # frames > 400 => long


def windows_per_clip(frames: int) -> int:
    return 2 if frames <= 200 else 3 if frames <= 400 else 4 if frames <= 1000 else 6


def deployed_window_starts(n: int, chunk: int = 14, overlap: int = 3) -> list[int]:
    """Exact start indices of inpainting_inference.main's window loop."""
    starts = []
    first = True
    for i in range(0, n, chunk - overlap):
        if i + overlap >= n:
            break
        if not first and i + chunk > n:
            starts.append(max(n + overlap - chunk, 0))
        else:
            starts.append(i)
        first = False
    return starts


def pick_windows(n: int, w: int) -> list[int]:
    grid = deployed_window_starts(n)
    if w >= len(grid):
        return grid
    # evenly spaced over the grid (deterministic; no RNG so the choice is stable across seeds)
    return [grid[round(k * (len(grid) - 1) / (w - 1))] for k in range(w)] if w > 1 else [grid[len(grid) // 2]]


def main():
    inv = json.load(open(INV))
    rng = random.Random(SEED)
    excluded = {}
    usable = {}
    for cid, v in sorted(inv.items()):
        if v.get("frames") is None:
            excluded[cid] = "unreadable"
        elif (v["h"], v["w"]) not in ((2160, 3840), (4400, 4400)):
            excluded[cid] = f"tile {v['h']}x{v['w']} too small for 576x1024 crop"
        else:
            usable[cid] = v
    fmt = lambda c: "2160" if usable[c]["h"] == 2160 else "4400"
    frames = lambda c: usable[c]["frames"]
    is_long = lambda c: frames(c) >= LONG_MIN

    # ---- test ----
    test = sorted(FORCED_TEST)
    for f in ("2160", "4400"):
        pool = sorted(c for c in usable if fmt(c) == f and usable[c]["gt"] and frames(c) <= SHORT_MAX
                      and c not in test and c != CONTINUITY)
        rng.shuffle(pool)
        need = N_TEST_PER_FMT - sum(1 for c in test if fmt(c) == f)
        test += pool[:need]
    test = sorted(test)

    # ---- dev ----
    dev = []
    for f in ("2160", "4400"):
        pool = sorted(c for c in usable if fmt(c) == f and frames(c) <= SHORT_MAX
                      and c not in test and c != CONTINUITY)
        rng.shuffle(pool)
        dev += pool[:N_DEV_SHORT_PER_FMT]
    pool = sorted(c for c in usable if is_long(c) and c not in test)
    rng.shuffle(pool)
    dev += pool[:N_DEV_LONG]
    dev = sorted(dev)

    # ---- train pool, stratified nested order ----
    train = sorted(c for c in usable if c not in test and c not in dev)
    strata = {}
    for c in train:
        strata.setdefault((fmt(c), "long" if is_long(c) else "short"), []).append(c)
    for k in strata:
        strata[k].sort()
        rng.shuffle(strata[k])
    # force the continuity clip to the front of its stratum
    k160 = (fmt(CONTINUITY), "long" if is_long(CONTINUITY) else "short")
    strata[k160].remove(CONTINUITY); strata[k160].insert(0, CONTINUITY)
    total = len(train)
    order = []
    taken = {k: 0 for k in strata}
    # largest-remainder interleave: at position p, the stratum whose quota (p+1)*share is most under-served
    for p in range(total):
        best = None
        for k, lst in strata.items():
            if taken[k] >= len(lst):
                continue
            deficit = (p + 1) * len(lst) / total - taken[k]
            if best is None or deficit > best[0] or (deficit == best[0] and k < best[1]):
                best = (deficit, k)
        k = best[1]
        order.append(strata[k][taken[k]]); taken[k] += 1
    assert order[0] == CONTINUITY or CONTINUITY in order[:CURVE_POINTS[0]], "0160 must be inside the 13-point"
    if order[0] != CONTINUITY:
        order.remove(CONTINUITY); order.insert(0, CONTINUITY)
    curve = {str(n): order[:n] for n in CURVE_POINTS}
    curve["all"] = order

    # ---- per-clip capture plan ----
    clips = {}
    for c, v in usable.items():
        w = windows_per_clip(v["frames"])
        clips[c] = {
            "fmt": fmt(c), "frames": v["frames"], "gt": bool(v["gt"]), "long": is_long(c),
            "role": "test" if c in test else "dev" if c in dev else "train",
            "windows": pick_windows(v["frames"], w),
            "path": f"video_data/splatting/{c}_splatting_results.mp4",
        }

    def summary(ids):
        return {"n": len(ids), "2160": sum(1 for c in ids if fmt(c) == "2160"),
                "4400": sum(1 for c in ids if fmt(c) == "4400"),
                "long": sum(1 for c in ids if is_long(c)),
                "windows": sum(len(clips[c]["windows"]) for c in ids)}

    out = {
        "version": "fulldata_v1", "seed": SEED,
        "inventory_sha256": hashlib.sha256(open(INV, "rb").read()).hexdigest()[:16],
        "rules": [l for l in __doc__.strip().splitlines()[__doc__.strip().splitlines().index("Rules"):]],
        "excluded": excluded,
        "test": test, "dev": dev, "continuity": CONTINUITY,
        "train_order": order,
        "curve": curve,
        "summary": {"test": summary(test), "dev": summary(dev), "train": summary(order),
                    **{f"curve_{k}": summary(v) for k, v in curve.items()}},
        "clips": clips,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print(json.dumps(out["summary"], indent=1)); print("test", test); print("dev", dev); print("curve13", curve["13"]); print("wrote", OUT)


if __name__ == "__main__":
    main()
