"""Deterministic held-out split + nested training stream for full-data distillation.
usage: python make_split.py <clip_inventory.json> <out.json>   (seed fixed = 0)
"""
import json, random, sys, hashlib
inv = json.load(open(sys.argv[1])); SEED = 0; SEG = 152; WIN = 14; STRIDE = 11
LEGACY_TEST = ["0042", "0204", "0301"]                 # keep for continuity with the 2026-09-13 tables
LEGACY_DISTILL = ["0160"] + [f"{i:04d}" for i in range(1, 13)]   # seen by light_lvl0_multiclip13_r2 -> never in a test set
ok = {k: v for k, v in inv.items() if "h" in v and v["h"] >= 2000}         # drops 0312 (broken) and 0362-0365 (960x1280)
short = {k: v for k, v in ok.items() if v["frames"] <= 300}
long_ = {k: v for k, v in ok.items() if v["frames"] > 300}
fmt = lambda k: "4400" if ok[k]["h"] == 4400 else "2160"
rng = random.Random(SEED)
pool = {f: sorted(k for k in short if fmt(k) == f and short[k]["gt"] and k not in LEGACY_TEST and k not in LEGACY_DISTILL) for f in ("4400", "2160")}
draw = lambda f, n: [pool[f].pop(pool[f].index(x)) for x in rng.sample(pool[f], n)]
test_lpips = sorted(LEGACY_TEST + draw("4400", 3) + draw("2160", 2))          # 4 x 4400 + 4 x 2160, all with GT
test_relmse = sorted(draw("4400", 6) + draw("2160", 6))                        # TF activation cache clips (never trained on)
test = set(test_lpips) | set(test_relmse)
segs = [{"clip": k, "frame_start": 0, "n_frames": min(short[k]["frames"], SEG), "fmt": fmt(k), "kind": "short"} for k in sorted(short) if k not in test]
for k in sorted(long_):
    K = min(4, max(1, long_[k]["frames"] // 500))
    for j in range(K):
        fs = 0 if K == 1 else round(j * (long_[k]["frames"] - SEG) / (K - 1))
        segs.append({"clip": k, "frame_start": fs, "n_frames": SEG, "fmt": fmt(k), "kind": "long"})
rng.shuffle(segs)
a = [s for s in segs if s["fmt"] == "4400"]; b = [s for s in segs if s["fmt"] == "2160"]
first = [s for s in b if s["clip"] == "0160" and s["frame_start"] == 0][0]; b.remove(first)
stream = [first]; ia = ib = 0
while ia < len(a) or ib < len(b):                       # alternate formats so every prefix is ~balanced
    if ia < len(a) and (ia * len(b) <= ib * len(a) or ib >= len(b)): stream.append(a[ia]); ia += 1
    else: stream.append(b[ib]); ib += 1
for i, s in enumerate(stream): s["idx"] = i; s["seed"] = 1234 + 7919 * i     # per-segment noise seed (eval keeps 1234)
n_win = lambda n: len([i for i in range(0, n, STRIDE) if i + 3 < n])
out = {"version": "heldout_v1", "seed": SEED, "inventory": sys.argv[1], "segment_frames": SEG, "window_frames": WIN, "window_stride": STRIDE,
       "excluded": {"broken": ["0312"], "too_small_960x1280": ["0362", "0363", "0364", "0365"], "long_clips_not_in_test": sorted(long_)},
       "legacy_distill_clips_never_tested": LEGACY_DISTILL,
       "test_lpips": test_lpips, "test_relmse": test_relmse,
       "eval_windows_relmse": [0, 6, 12], "eval_crop": "center", "eval_seed": 1234,
       "milestones": [13, 40, 120, len(stream)], "train_stream": stream,
       "counts": {"short_train": sum(s["kind"] == "short" for s in stream), "long_segments": sum(s["kind"] == "long" for s in stream),
                  "train_4400": sum(s["fmt"] == "4400" for s in stream), "train_2160": sum(s["fmt"] == "2160" for s in stream),
                  "windows_per_segment_full": n_win(SEG)}}
js = json.dumps(out, indent=1); open(sys.argv[2], "w").write(js)
print("sha256", hashlib.sha256(js.encode()).hexdigest()); print(json.dumps({k: out[k] for k in ("test_lpips", "test_relmse", "milestones", "counts")}))
print("first 40 stream:", [(s["clip"], s["frame_start"], s["fmt"][0]) for s in stream[:40]])
