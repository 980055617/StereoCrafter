"""Compare the skeptic's own ROW lines (scores_skeptic_{0301,0204}.txt) with every ROW line the two controls reported.
Match on the render path; report any difference in lpips (>5e-5), sharp (>5e-6), alignment offset (dy,dx), or n."""
import glob, os, re, sys
REPO = "/home/kawa/master_project/StereoCrafter"; os.chdir(REPO)
SK = "scripts/distill/runs/clean_controls/skeptic"
def rows(path):
    out = {}
    for line in open(path):
        if not line.startswith("ROW "): continue
        d = dict(kv.split("=", 1) for kv in line.strip().split()[1:])
        out[d["path"]] = d
    return out
mine = {}
for c in ("0301", "0204"): mine.update(rows(f"{SK}/scores_skeptic_{c}.txt"))
reported = {}
for f in sorted(glob.glob("scripts/distill/runs/clean_controls/selfA/scores_*.txt") + glob.glob("scripts/distill/runs/clean_controls/regB/scores/scores_*.txt")):
    if "regdiag" in f: continue          # registered-GT diagnostic rows use a different target; not the standard number
    for p, d in rows(f).items():
        reported.setdefault(p, []).append((f, d))
print(f"my rows: {len(mine)}   reported render paths: {len(reported)}")
offs = set(); ns = set(); nmatch = 0; ndiff = 0; unmatched = []
for p, d in sorted(mine.items()):
    offs.add((d["clip"], d["dy"], d["dx"])); ns.add(d["n"])
    if p not in reported:
        unmatched.append(p); continue
    for f, r in reported[p]:
        dl = abs(float(d["lpips"]) - float(r["lpips"])); ds = abs(float(d["sharp"]) - float(r["sharp"]))
        same_off = (d["dy"], d["dx"], d["n"]) == (r["dy"], r["dx"], r["n"])
        ok = dl <= 5e-5 and ds <= 5e-6 and same_off
        nmatch += ok; ndiff += (not ok)
        flag = "OK  " if ok else "DIFF"
        print(f"{flag} {d['tag'][:40]:40s} mine lpips={float(d['lpips']):.4f} sharp={float(d['sharp']):.4f} ({d['dy']},{d['dx']}) n={d['n']} | reported {float(r['lpips']):.4f} {float(r['sharp']):.4f} ({r['dy']},{r['dx']}) n={r['n']}  [{os.path.basename(f)}]")
print(f"\nmatched {nmatch} (file,row) pairs, {ndiff} differences")
print("renders I scored that no control reported a standard row for:", [os.path.basename(os.path.dirname(p)) for p in unmatched])
print("distinct (clip,dy,dx) alignment offsets across all my rows:", sorted(offs), "; distinct n:", sorted(ns))
for p in sorted(reported):
    if p not in mine: print("reported but not re-scored by me:", p)
