"""Collate the mech/ TEST A runs: per-sigma loss early vs late, pre-clip grad-norm trajectory, relative weight change,
and the pre-training diagnostics.  usage: python summarize_mech.py <run_dir> [<run_dir> ...]"""
import os, sys, json, csv, glob
import statistics as st

print(f"{'run':10s} {'g0_mean':>9s} {'gn_mean':>9s} {'gn_first':>9s} {'gn_max':>9s} {'rel_dW/W':>10s} {'loss_1st50':>12s} {'loss_last50':>12s}")
for d in sys.argv[1:]:
    name = os.path.basename(d.rstrip('/'))
    rows = list(csv.DictReader(open(os.path.join(d, "train_log.csv"))))
    gn = [float(r["grad_norm_preclip"]) for r in rows]; ls = [float(r["loss"]) for r in rows]
    meta = json.load(open(os.path.join(d, "meta.json")))
    post = json.load(open(os.path.join(d, "post.json"))) if os.path.exists(os.path.join(d, "post.json")) else {"rel_dw": float('nan')}
    print(f"{name:10s} {meta['g0_mean']:9.4g} {st.mean(gn):9.4g} {gn[0]:9.4g} {max(gn):9.4g} "
          f"{post['rel_dw']:10.6f} {st.mean(ls[:50]):12.6g} {st.mean(ls[-50:]):12.6g}")

for d in sys.argv[1:]:
    name = os.path.basename(d.rstrip('/'))
    rows = list(csv.DictReader(open(os.path.join(d, "train_log.csv"))))
    n = len(rows); half = n // 2
    print(f"\n--- {name}: per-trajectory-step loss and pre-clip grad norm, first half vs second half of the {n} steps ---")
    print(f"{'traj k':>7s} {'sigma':>10s} {'n':>4s} {'loss 1st':>12s} {'loss 2nd':>12s} {'gn 1st':>10s} {'gn 2nd':>10s}")
    for k in sorted({int(r["traj_step"]) for r in rows}):
        a = [r for i, r in enumerate(rows) if int(r["traj_step"]) == k and i < half]
        b = [r for i, r in enumerate(rows) if int(r["traj_step"]) == k and i >= half]
        f = lambda rs, key: st.mean([float(r[key]) for r in rs]) if rs else float('nan')
        print(f"{k:>7d} {float(a[0]['sigma']) if a else float(b[0]['sigma']):>10.4g} {len(a)+len(b):>4d} "
              f"{f(a,'loss'):>12.6g} {f(b,'loss'):>12.6g} {f(a,'grad_norm_preclip'):>10.4g} {f(b,'grad_norm_preclip'):>10.4g}")
    meta = json.load(open(os.path.join(d, "meta.json")))
    print(f"  pre-training target diagnostics (window {meta['windows'][0]}):")
    for r in meta["pre_diag"]:
        print("    " + "  ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()))
    print("  numerics floor (rel diff vs the training-context batch-1 forward):")
    for r in meta["numerics_floor"]:
        print(f"    w{r['w']} step{r['k']}: repeat {r['repeat_rel']:.3e}  deployed-context batch-1 {r['deploy_ctx_batch1_rel']:.3e}  "
              f"deployed CFG batch-2 cond half {r['cfg_batch2_vs_batch1_rel']:.3e}")
