"""Per-sigma mean loss at steps 1-50 vs 251-300, s/step, peak memory from minift/<variant>/train_log.csv.
usage: python summarize_minift.py <run_dir> [<run_dir> ...]"""
import sys, csv, os, json, statistics as _st
class st:  # tolerate empty blocks (partial CSV of a running job)
    mean=staticmethod(lambda it: (lambda l: _st.mean(l) if l else float("nan"))(list(it))); median=staticmethod(lambda it: (lambda l: _st.median(l) if l else float("nan"))(list(it)))
for d in sys.argv[1:]:
    rows = list(csv.DictReader(open(os.path.join(d, "train_log.csv"))))
    meta = json.load(open(os.path.join(d, "meta.json")))
    SIG8 = meta["sigmas"]; import math
    def band(x):   # continuous sigma (lognormal runs) -> nearest deployment sigma in log space
        return x if x in SIG8 else min(SIG8, key=lambda s: abs(math.log(s) - math.log(x)))
    for r in rows: r["sigma"] = band(float(r["sigma"]))
    sig = sorted({float(r["sigma"]) for r in rows}, reverse=True)
    if len({float(r0) for r0 in [row["sigma"] for row in rows]}) and meta.get("sigma_spec", "deploy8").startswith("lognormal"): print(f"   (sigma_spec={meta['sigma_spec']}: continuous sigmas binned to the nearest deployment sigma in log space)")
    def block(lo, hi): return [r for r in rows if lo <= int(r["step"]) <= hi]
    early, late = block(1, 50), block(251, 300)
    print(f"== {d} variant={meta['variant']} sigma_spec={meta.get('sigma_spec','deploy8')} steps={len(rows)} windows={len(meta['windows'])} mask_frac min/mean={min(meta['mask_fracs']):.4f}/{st.mean(meta['mask_fracs']):.4f}")
    print(f"   s/step mean={st.mean(float(r['step_s']) for r in rows):.3f} (median {st.median(float(r['step_s']) for r in rows):.3f}); peak alloc {max(float(r['peak_alloc_GiB']) for r in rows):.2f} GiB; "
          f"grad_norm preclip mean all/early/late = {st.mean(float(r['grad_norm_preclip']) for r in rows):.3f}/{st.mean(float(r['grad_norm_preclip']) for r in early):.3f}/{st.mean(float(r['grad_norm_preclip']) for r in late):.3f}, clipped(>1) {sum(float(r['grad_norm_preclip'])>1 for r in rows)}/{len(rows)}")
    print(f"   {'sigma':>8s} {'n_early':>7s} {'loss_1-50':>10s} {'n_late':>6s} {'loss_251-300':>12s} {'delta':>8s}")
    for s in sig:
        e = [float(r["loss"]) for r in early if float(r["sigma"]) == s]; l = [float(r["loss"]) for r in late if float(r["sigma"]) == s]
        me = st.mean(e) if e else float("nan"); ml = st.mean(l) if l else float("nan")
        print(f"   {s:8.4g} {len(e):7d} {me:10.4f} {len(l):6d} {ml:12.4f} {ml-me:+8.4f}")
    print(f"   all-sigma mean loss 1-50 = {st.mean(float(r['loss']) for r in early):.4f}, 251-300 = {st.mean(float(r['loss']) for r in late):.4f}")
