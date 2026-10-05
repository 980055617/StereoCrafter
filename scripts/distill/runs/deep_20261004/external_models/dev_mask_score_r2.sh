#!/bin/bash
# DEV-only mask-threshold check (PREREG_ADDENDUM_r2_dev_mask.txt): waits for RENDER_DEV_T95, then UNREG LPIPS (CPU copy of
# score_clip_ll.py) of primary vs t95 on 0040 0082 0091, then applies the addendum's selection rule.
set -u
cd /home/kawa/master_project/StereoCrafter
set +u; source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate stereocrafter; set -u
export CUDA_VISIBLE_DEVICES="" SCORE_STEP=4 NTHREADS=8
R=scripts/distill/runs/deep_20261004/external_models
O=outputs/deep_20261004/external_models
OUT=$R/DEV_MASK_UNREG_r2.txt
until grep -q "RENDER_DEV_T95" $R/chain_m2svid_fa_w16_r2.log 2>/dev/null; do sleep 15; done
[ -e $OUT ] && { echo "exists $OUT"; exit 1; }
SPECS=()
for c in 0040 0082 0091; do
  for t in m2svid_fa_w16 m2svid_fa_w16_t95; do
    M=$O/clips/${c}_${t}_ll/${c}_inpainting_results_sbs.mkv
    [ -s $M ] && SPECS+=("$c=$M")
  done
done
python $R/score_clip_ll_cpu_r2.py "${SPECS[@]}" > $OUT 2>&1
python - "$OUT" >> $OUT 2>&1 <<'EOF'
import sys
rows = {}
for line in open(sys.argv[1]):
    if line.startswith("ROW "):
        kv = dict(t.split("=", 1) for t in line.split()[1:])
        rows[(kv["clip"], kv["tag"].split("_", 1)[1])] = float(kv["lpips"])
clips = sorted({c for c, _ in rows})
pairs = [(c, rows[(c, "m2svid_fa_w16_ll")], rows[(c, "m2svid_fa_w16_t95_ll")]) for c in clips
         if (c, "m2svid_fa_w16_ll") in rows and (c, "m2svid_fa_w16_t95_ll") in rows]
for c, p, v in pairs:
    print(f"DEV {c}: primary {p:.6f}  t95 {v:.6f}  t95-primary {v - p:+.6f}")
if pairs:
    d = sum(v - p for _, p, v in pairs) / len(pairs)
    w = sum(v < p for _, p, v in pairs)
    sel = d <= -0.005 and w >= 2 and len(pairs) == 3
    print(f"DEV_RULE mean(t95-primary) {d:+.6f}, t95 better on {w}/{len(pairs)} -> "
          f"{'SELECT t95' if sel else 'KEEP primary'}")
EOF
echo "DEV_MASK_DONE $(date '+%F_%T')" >> $OUT
