#!/bin/bash
# SLOT BUDGET - PASS A: total UNet-forward time + peak VRAM, reproducing the published
# methodology byte for byte (the provenance of -5.3 / -20.5 / -21.7 % in
# scripts/distill/runs/fulldata/FINAL_TABLES.txt is scripts/distill/bench2.py, driven by
# scripts/distill/fulldata_tail.sh (576x1024, n=3 -> bench_tail.txt) and
# scripts/distill/fullhd.sh / fixup_lane.sh (1024x1792 + 1024x1920, n=2 -> bench_fullhd.txt)).
#
# bench2.py is used UNMODIFIED. Same env as published: BS=2 (guidance 1.01 keeps the
# CFG-doubled batch), GATE=1.0 DS=128 EXP=1 BIDIR=fwd, EXCLUDE=__nomatch__, no CKPT
# (the published bench timed the architecture, not trained weights), 14 frames,
# 3 untimed warmup forwards then 10 timed forwards averaged, exclusive GPU.
#
# Configs: origin (no replacement) / mamba5 (the shipped deliverable) / mamba2 (down_blocks.0 only).
# Loop order reps-outer, configs-inner (as published) so thermal drift cannot alias onto a config.
# Resolutions are stated as HxW: h576w1024, h1024w1792, h1024w1920.
set -u
cd /home/kawa/master_project/StereoCrafter
D=scripts/distill
R=$D/runs/slotbudget
OUT=$R/bench_totals_v1.txt
export CUDA_VISIBLE_DEVICES=0
echo "BENCH_TOTALS_V1_START $(date +%F_%H:%M:%S) gpu=$CUDA_VISIBLE_DEVICES" | tee -a "$OUT"
nvidia-smi --query-gpu=index,name,memory.used,temperature.gpu --format=csv,noheader | tee -a "$OUT"

run_one() {  # label include H W rep
  local L=$1 INC=$2 H=$3 W=$4 rep=$5
  H=$H W=$W BS=2 GATE=1.0 DS=128 EXP=1 BIDIR=fwd \
    python3 $D/bench2.py "${L}_h${H}w${W}_r${rep}" "$INC" "__nomatch__" \
    > "$R/logs/bench_${L}_h${H}w${W}_r${rep}.log" 2>&1
  grep -E '^RESULT' "$R/logs/bench_${L}_h${H}w${W}_r${rep}.log" | tee -a "$OUT"
  # provenance: which slots the adapter actually replaced
  grep -cE '^\[MambaAdapter\]\[self-attn\] replaced|total_replaced' "$R/logs/bench_${L}_h${H}w${W}_r${rep}.log" >/dev/null
}

# reps 1-3: all three resolutions.  reps 4-5: the deployed resolution only (cheap, and the
# smallest absolute effect, so it gets the most samples).
for rep in 1 2 3; do
  for V in "origin:__nomatch__" "mamba5:down_blocks.0.*,up_blocks.3.*" "mamba2:down_blocks.0.*"; do
    L=${V%%:*}; INC=${V#*:}
    run_one "$L" "$INC" 576 1024 "$rep"
    run_one "$L" "$INC" 1024 1792 "$rep"
    run_one "$L" "$INC" 1024 1920 "$rep"
  done
  echo "  rep $rep done $(date +%H:%M:%S)" | tee -a "$OUT"
done
for rep in 4 5; do
  for V in "origin:__nomatch__" "mamba5:down_blocks.0.*,up_blocks.3.*" "mamba2:down_blocks.0.*"; do
    L=${V%%:*}; INC=${V#*:}
    run_one "$L" "$INC" 576 1024 "$rep"
  done
  echo "  rep $rep (576 only) done $(date +%H:%M:%S)" | tee -a "$OUT"
done
echo "BENCH_TOTALS_V1_DONE $(date +%F_%H:%M:%S)" | tee -a "$OUT"
