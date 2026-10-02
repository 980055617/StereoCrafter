#!/bin/bash
# v2 of bench_deliv_v1.sh: the config list is a parameter, so the two checkpoint-free/shipped rows can
# run in GPU 0's post-training window before the deliverable exists, and the deliverable row can be
# appended to the SAME output file afterwards.  Methodology is byte-identical to
# scripts/distill/runs/slotbudget/bench_totals_v1.sh (which is the provenance of the published
# -5.3 / -20.5 / -21.7 %): scripts/distill/bench2.py UNMODIFIED, BS=2 (guidance 1.01 keeps the
# CFG-doubled batch), 14 frames, fp16, GATE=1.0 DS=128 EXP=1 BIDIR=fwd, EXCLUDE=__nomatch__,
# 3 untimed warmup forwards then 10 timed forwards / 10 with cuda.synchronize bracketing,
# peak = max_memory_allocated reset after warmup, ONE PROCESS PER REPEAT, reps-outer/configs-inner,
# EXCLUSIVE GPU.  Resolutions are HxW: h576w1024 (deployed) / h1024w1792 / h1024w1920.
# Unlike the published bench (no CKPT -> it timed the ARCHITECTURE), the mamba rows here load real
# weights so the deliverable's speed row is a PRIMARY measurement of the shipped file.
# usage: bench_deliv_v2.sh <gpu> <out.txt> <deliverable_or_-> <cfg...>      cfg in origin|mamba5ck|mamba5deliv
set -u
cd /home/kawa/master_project/StereoCrafter
GPU=$1; OUT=$2; DELIV=$3; shift 3; CFGS="$*"
PY=/home/kawa/miniconda3/envs/stereocrafter/bin/python
DD=scripts/distill
R=$DD/runs/beyond_distil_mamba_scaled
SHIP=/mnt/ssd_data/stereocrafter_weights/_distill_injected/light_lvl0_fulldata333_v2_8k_mamba_only.pt
MI='down_blocks.0.*,up_blocks.3.*'
export CUDA_VISIBLE_DEVICES=$GPU
mkdir -p $R/logs
echo "BENCH_V2_START $(date +%F_%H:%M:%S) gpu=$GPU cfgs=[$CFGS] deliv=$DELIV" | tee -a "$OUT"
nvidia-smi --query-gpu=index,name,memory.used,temperature.gpu --format=csv,noheader | tee -a "$OUT"
md5sum "$SHIP" | tee -a "$OUT"
[ "$DELIV" != "-" ] && md5sum "$DELIV" | tee -a "$OUT"

run_one() {  # cfg H W rep
  local L=$1 H=$2 W=$3 rep=$4 INC CK
  case $L in
    origin)      INC='__nomatch__'; CK="" ;;
    mamba5ck)    INC="$MI";         CK="$SHIP" ;;
    mamba5deliv) INC="$MI";         CK="$DELIV" ;;
    *) echo "unknown cfg $L"; return 1 ;;
  esac
  local LF="$R/logs/bench_${L}_h${H}w${W}_r${rep}.log"
  if [ -s "$LF" ] && grep -q '^RESULT' "$LF"; then
    echo "  SKIP $L h${H}w${W} r$rep (already measured)" ; grep -E '^RESULT' "$LF" >> "$OUT"; return 0
  fi
  if [ -n "$CK" ]; then
    H=$H W=$W BS=2 GATE=1.0 DS=128 EXP=1 BIDIR=fwd CKPT="$CK" \
      $PY $DD/bench2.py "${L}_h${H}w${W}_r${rep}" "$INC" "__nomatch__" > "$LF" 2>&1
  else
    H=$H W=$W BS=2 GATE=1.0 DS=128 EXP=1 BIDIR=fwd \
      $PY $DD/bench2.py "${L}_h${H}w${W}_r${rep}" "$INC" "__nomatch__" > "$LF" 2>&1
  fi
  grep -E '^RESULT' "$LF" | tee -a "$OUT"
}

for rep in 1 2 3; do
  for HW in "576 1024" "1024 1792" "1024 1920"; do
    set -- $HW
    for L in $CFGS; do run_one $L $1 $2 $rep; done
  done
  echo "  rep $rep done $(date +%H:%M:%S)" | tee -a "$OUT"
done
# reps 4-5 at the DEPLOYED resolution only (smallest absolute effect -> most samples, as published n=5)
for rep in 4 5; do
  for L in $CFGS; do run_one $L 576 1024 $rep; done
  echo "  rep $rep (576 only) done $(date +%H:%M:%S)" | tee -a "$OUT"
done
echo "BENCH_V2_DONE $(date +%F_%H:%M:%S) cfgs=[$CFGS]" | tee -a "$OUT"
