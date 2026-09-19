"""Capture manifests for fulldata_v1: chunk files of <= CHUNK windows in capture order, plus prefix markers.
order: dev -> test -> curve-13 protocol windows -> curve-13 control windows (14-window grid, control-only data)
       -> remaining train_order windows (curve 40 -> 120 -> all prefixes).
usage: python make_manifest.py [CHUNK=24]  -> scripts/distill/runs/fulldata/manifests/cap_XXX.json + prefixes.json
"""
import json, os, sys, hashlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_split_fulldata import deployed_window_starts, pick_windows
HERE = os.path.dirname(os.path.abspath(__file__)); SPLIT = os.path.join(HERE, "splits", "fulldata_v1.json")
OUTD = os.path.join(HERE, "runs", "fulldata", "manifests"); os.makedirs(OUTD, exist_ok=True)
CHUNK = int(sys.argv[1]) if len(sys.argv) > 1 else 24
s = json.load(open(SPLIT)); C = s["clips"]; sha = hashlib.sha256(open(SPLIT, "rb").read()).hexdigest()[:16]
entries = []; groups = {}
def add(clip, start, role, group):
    idx = len(entries)
    entries.append({"idx": idx, "clip": clip, "path": C[clip]["path"], "start": int(start), "seed": 1234 + 7919 * idx,
                    "fmt": C[clip]["fmt"], "frames": C[clip]["frames"], "role": role, "group": group})
    groups.setdefault(group, []).append(idx)
for c in s["dev"]:
    for w in C[c]["windows"]: add(c, w, "dev", "dev")
for c in s["test"]:
    for w in C[c]["windows"]: add(c, w, "test", "test")
c13 = s["curve"]["13"]
for c in c13:
    for w in C[c]["windows"]: add(c, w, "train", "c13")
for c in c13:                                  # control: fill up to a 14-window grid per clip
    grid = deployed_window_starts(C[c]["frames"]); want = pick_windows(C[c]["frames"], 14) if len(grid) > 14 else grid
    for w in want:
        if w not in C[c]["windows"]: add(c, w, "train", "c13w14")
seen13 = set(c13)
for c in s["train_order"]:
    if c in seen13: continue
    g = "c40" if c in s["curve"]["40"] else ("c120" if c in s["curve"]["120"] else "call")
    for w in C[c]["windows"]: add(c, w, "train", g)
# chunk without splitting a clip across files
chunks = []; cur = []
for e in entries:
    if cur and (e["group"] != cur[-1]["group"] or (len(cur) >= CHUNK and e["clip"] != cur[-1]["clip"])): chunks.append(cur); cur = []   # never mix groups in a chunk
    cur.append(e)
if cur: chunks.append(cur)
for k, ch in enumerate(chunks):
    json.dump({"split_sha": sha, "chunk": k, "entries": ch}, open(os.path.join(OUTD, f"cap_{k:03d}.json"), "w"))
# prefix -> last chunk index that completes it
def last_chunk(idxs): 
    m = max(idxs); return next(k for k, ch in enumerate(chunks) if any(e["idx"] == m for e in ch))
pref = {"dev": last_chunk(groups["dev"]), "test": last_chunk(groups["test"]), "c13": last_chunk(groups["c13"]), "c13w14": last_chunk(groups["c13w14"]),
        "c40": last_chunk(groups["c40"]), "c120": last_chunk(groups["c120"]), "all": len(chunks) - 1}
json.dump({"split_sha": sha, "chunk_size": CHUNK, "n_windows": len(entries), "n_chunks": len(chunks), "groups": {g: len(v) for g, v in groups.items()},
           "prefix_last_chunk": pref}, open(os.path.join(OUTD, "prefixes.json"), "w"), indent=1)
print(f"windows={len(entries)} chunks={len(chunks)} groups={ {g: len(v) for g, v in groups.items()} } prefix_last_chunk={pref}")
