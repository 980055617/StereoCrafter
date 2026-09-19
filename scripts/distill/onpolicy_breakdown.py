"""Per-slot, per-denoising-step student-vs-teacher relative MSE from an on-policy cache.
usage: python onpolicy_breakdown.py <cache_dir>   (records must carry y_student)
Step 0 of every window feeds the SAME latent to student and teacher (pure noise + conditioning),
so at step 0 the first slot (down0.attn0) sees exactly the teacher-forced input distribution:
its step-0 error == teacher-forced error  => numerics faithful; error growing with step => cascade drift.
"""
import sys, glob, os, collections, torch
C = sys.argv[1]
acc = collections.defaultdict(lambda: [0.0, 0.0])
for f in sorted(glob.glob(os.path.join(C, "*__call*.pt"))):
    r = torch.load(f, map_location="cpu")
    if r.get("y_student") is None: continue
    step = r["call"] % 8; slot = r["name"].split(".transformer")[0]
    a = acc[(slot, step)]; a[0] += (r["y_student"].float() - r["y"].float()).pow(2).sum().item(); a[1] += r["y"].float().pow(2).sum().item()
slots = sorted({k[0] for k in acc}, key=lambda s: (s.startswith("up"), s))
print(f"{'slot':28s} " + " ".join(f"step{s:d}" for s in range(8)) + "   all")
for s in slots:
    row = [acc[(s, st)][0] / max(acc[(s, st)][1], 1e-12) for st in range(8)]
    tot = sum(acc[(s, st)][0] for st in range(8)) / max(sum(acc[(s, st)][1] for st in range(8)), 1e-12)
    print(f"{s:28s} " + " ".join(f"{v:5.3f}" for v in row) + f"   {tot:5.3f}")
