"""CONTROL A post-hoc weight analysis.
rel ||dW||/||W|| of the 15 trained up_blocks.3 attn1 tensors at steps 100/200/300, measured against the TRUE bf16 start point
(start_bf16.pt written by xcheck_mini_ft_ll.py, cross-checked bitwise against the fp16 safetensors cast to bf16), for the lossless
run AND for the old mp4v P1-null run (minift/null) so the two are like-for-like.  minift/deltas_bisect.txt measured the old run
against the fp16 FILE, which puts the bf16 quantisation of the start point into the "delta"; that floor is reported here too.
Also: cosine(lossless delta, old delta) per step, per-tensor share, and train_log.csv statistics (step-1 pre-clip grad norm,
mean grad norm, per-sigma loss early/late) for both runs.
usage: python deltas_ll.py <llnull_dir> [<old_null_dir>]
"""
import os, sys, glob, csv, json, math, statistics, torch, torch.nn.functional as F
from safetensors import safe_open
REPO = "/home/kawa/master_project/StereoCrafter"; os.chdir(REPO)
NEW = sys.argv[1]; OLD = sys.argv[2] if len(sys.argv) > 2 else "scripts/distill/runs/diag_trainer/minift/null"
start = torch.load(os.path.join(NEW, "start_bf16.pt"), map_location="cpu"); KEYS = sorted(start)
assert len(KEYS) == 15 and all(v.dtype == torch.bfloat16 for v in start.values()), (len(KEYS), {v.dtype for v in start.values()})
f16 = {}
with safe_open(glob.glob("weights/StereoCrafter/unet_diffusers/*.safetensors")[0], "pt", device="cpu") as sf:
    for k in KEYS: f16[k] = sf.get_tensor(k)
print("safetensors dtype:", {v.dtype for v in f16.values()})
cat = lambda d, fn=lambda x: x.float(): torch.cat([fn(d[k]).flatten() for k in KEYS])
W_bf16 = cat(start); W_f16 = cat(f16); W_f16_as_bf16 = cat(f16, lambda x: x.to(torch.bfloat16).float())
print(f"start_bf16.pt vs safetensors(fp16)->bf16 : maxabs diff = {(W_bf16 - W_f16_as_bf16).abs().max().item():.3e}  (0 => the trainer's start point IS the bf16 cast of the fp16 file)")
print(f"bf16 quantisation floor of the start point : rel ||bf16 - fp16|| / ||W|| = {((W_bf16 - W_f16).norm() / W_bf16.norm()).item():.5f}   <- what the old fp16-file measurement folded into 'dW'")
def load(d, step): return cat(torch.load(os.path.join(d, f"step{step}.pt"), map_location="cpu"))
print(f"\n{'run':10s} {'step':>5s} {'rel dW/W vs bf16 start':>24s} {'rel dW/W vs fp16 file (old method)':>36s} {'cos(new,old) same step':>24s}")
deltas = {}
for step in (100, 200, 300):
    for name, d in (("llnull", NEW), ("old null", OLD)):
        Wk = load(d, step); deltas[(name, step)] = Wk - W_bf16
    for name in ("llnull", "old null"):
        dd = deltas[(name, step)]; Wk = dd + W_bf16
        cos = F.cosine_similarity(deltas[("llnull", step)], deltas[("old null", step)], dim=0).item()
        print(f"{name:10s} {step:5d} {(dd.norm() / W_bf16.norm()).item():24.5f} {((Wk - W_f16).norm() / W_f16.norm()).item():36.5f} {cos:24.3f}")
print("\nper-tensor share of ||dW||^2 at step 300 (llnull / old null):")
off = 0
for k in KEYS:
    n = start[k].numel(); a = deltas[("llnull", 300)][off:off + n].pow(2).sum() / deltas[("llnull", 300)].pow(2).sum(); b = deltas[("old null", 300)][off:off + n].pow(2).sum() / deltas[("old null", 300)].pow(2).sum()
    print(f"  {k.split('attentions.')[1]:40s} {a.item():.3f} / {b.item():.3f}"); off += n
print("\ntrain_log.csv statistics:")
for name, d in (("llnull", NEW), ("old null", OLD)):
    rows = list(csv.DictReader(open(os.path.join(d, "train_log.csv")))); meta = json.load(open(os.path.join(d, "meta.json")))
    gn = [float(r["grad_norm_preclip"]) for r in rows]; loss = [float(r["loss"]) for r in rows]
    print(f"== {name} ({d}) target={meta['target']}")
    print(f"   step-1 pre-clip grad norm = {gn[0]:.4f} (sigma {rows[0]['sigma']}, window {rows[0]['win_start']}) | mean over 300 steps = {statistics.mean(gn):.4f} | steps 1-50 {statistics.mean(gn[:50]):.4f} | 251-300 {statistics.mean(gn[250:]):.4f} | clipped(>1) {sum(g > 1 for g in gn)}/300")
    print(f"   loss mean 1-50 = {statistics.mean(loss[:50]):.4f} | 251-300 = {statistics.mean(loss[250:]):.4f} | s/step median {statistics.median(float(r['step_s']) for r in rows):.3f} | peak alloc {max(float(r['peak_alloc_GiB']) for r in rows):.2f} GiB")
    print(f"   {'sigma':>8s} {'n':>3s} {'loss_1-150':>10s} {'loss_151-300':>12s} {'gn_mean':>8s}")
    for s in sorted({float(r["sigma"]) for r in rows}, reverse=True):
        e = [float(r["loss"]) for r in rows[:150] if float(r["sigma"]) == s]; l = [float(r["loss"]) for r in rows[150:] if float(r["sigma"]) == s]; g = [float(r["grad_norm_preclip"]) for r in rows if float(r["sigma"]) == s]
        print(f"   {s:8.4g} {len(e) + len(l):3d} {statistics.mean(e) if e else float('nan'):10.4f} {statistics.mean(l) if l else float('nan'):12.4f} {statistics.mean(g):8.4f}")
# sigma/eps sequence identity check between the two runs (same seed -> same (win, sigma) schedule)
rn = list(csv.DictReader(open(os.path.join(NEW, "train_log.csv")))); ro = list(csv.DictReader(open(os.path.join(OLD, "train_log.csv"))))
same = all(a["win_start"] == b["win_start"] and a["sigma"] == b["sigma"] for a, b in zip(rn, ro)) and len(rn) == len(ro) == 300
print(f"\n(window, sigma) schedule identical between the two runs for all 300 steps: {same}")
print("DELTAS_DONE")
