#!/bin/bash
# finalcheck_20261004 / bench lane, driver v1.
# Reproduces scripts/distill/runs/slotbudget/bench_totals_v1.sh (the 2026-10-01 lane) with scripts/distill/bench2.py
# UNMODIFIED, adding BS=1 (guidance 1.00) next to BS=2 (guidance 1.01).  See PREREG.txt in this directory.
#
# usage: bench_v1.sh OUTDIR smoke|full
#   smoke : origin + mamba5 at BS=1, 576x1024, rep 0 (never counted)
#   full  : reps 1..3 outer, BS (2,1), config (origin, mamba5), res (576x1024, 1024x1792, 1024x1920) inner
# Must run inside the conda env stereocrafter (its activate hook puts the repo root on PYTHONPATH).
set -u
cd /home/kawa/master_project/StereoCrafter
OUTDIR=$1; MODE=$2
D=scripts/distill
mkdir -p "$OUTDIR/logs"
RES_TXT="$OUTDIR/bench_results.txt"      # RESULT lines (verbatim from bench2.py) + RUNMETA lines
CHK="$OUTDIR/exclusivity_checks.txt"     # nvidia-smi snapshots before/after every process
GPULOG="$OUTDIR/gpu_monitor.csv"         # 2 s sampler over the whole bench

export CUDA_VISIBLE_DEVICES=0
unset PYTORCH_CUDA_ALLOC_CONF
for v in $(env | grep -oE '^MAMBA_[A-Za-z0-9_]+'); do unset "$v"; done
env | sort > "$OUTDIR/env_at_start.txt"
md5sum $D/bench2.py > "$OUTDIR/bench2_md5.txt"
python3 -c "import torch, diffusers, sys; print('python', sys.version.split()[0], 'torch', torch.__version__, 'cuda', torch.version.cuda, 'diffusers', diffusers.__version__, 'gpu', torch.cuda.get_device_name(0))" > "$OUTDIR/versions.txt" 2>&1

nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,temperature.gpu,clocks.sm,power.draw \
  --format=csv,noheader -l 2 > "$GPULOG" 2>&1 &
MON=$!
trap 'kill $MON 2>/dev/null' EXIT

echo "BENCH_START $(date +%F_%H:%M:%S) mode=$MODE outdir=$OUTDIR" | tee -a "$RES_TXT"
nvidia-smi --query-gpu=index,name,memory.used,temperature.gpu --format=csv,noheader | tee -a "$RES_TXT"

snap() {  # tag phase
  {
    echo "$2 $1 $(date +%F_%H:%M:%S)"
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,temperature.gpu,clocks.sm --format=csv,noheader
    echo "compute-apps:"
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
  } >> "$CHK" 2>&1
}

attempt() {  # L INC H W BS tag  -> returns 0 iff rc=0 and a RESULT line exists
  local L=$1 INC=$2 H=$3 W=$4 BS=$5 tag=$6
  local log="$OUTDIR/logs/bench_${tag}.log"
  snap "$tag" PRE
  local t0; t0=$(date +%s.%N)
  H=$H W=$W BS=$BS GATE=1.0 DS=128 EXP=1 BIDIR=fwd \
    python3 $D/bench2.py "$tag" "$INC" "__nomatch__" > "$log" 2>&1
  local rc=$?
  local t1; t1=$(date +%s.%N)
  snap "$tag" POST
  local res; res=$(grep -E '^RESULT' "$log" | tail -1)
  local repl; repl=$(grep -oE 'total_replaced=[0-9]+' "$log" | tail -1)
  [ -n "$res" ] && echo "$res" | tee -a "$RES_TXT"
  echo "RUNMETA tag=$tag config=$L include=$INC H=$H W=$W BS=$BS rc=$rc ${repl:-total_replaced=NA} start=$t0 end=$t1 has_result=$([ -n "$res" ] && echo 1 || echo 0)" | tee -a "$RES_TXT"
  [ $rc -eq 0 ] && [ -n "$res" ]
}

run_one() {  # L INC H W BS rep : one process, retried at most once on failure
  local L=$1 INC=$2 H=$3 W=$4 BS=$5 rep=$6
  local tag="${L}_bs${BS}_h${H}w${W}_r${rep}"
  if ! attempt "$L" "$INC" "$H" "$W" "$BS" "$tag"; then
    echo "FAILED $tag -> retry once; last lines of the log:" | tee -a "$RES_TXT"
    tail -5 "$OUTDIR/logs/bench_${tag}.log" | sed 's/^/    /' | tee -a "$RES_TXT"
    attempt "$L" "$INC" "$H" "$W" "$BS" "${tag}_retry" || echo "FAILED_TWICE $tag" | tee -a "$RES_TXT"
  fi
}

if [ "$MODE" = "smoke" ]; then
  run_one origin "__nomatch__" 576 1024 1 0
  run_one mamba5 "down_blocks.0.*,up_blocks.3.*" 576 1024 1 0
else
  for rep in 1 2 3; do
    for BS in 2 1; do
      for V in "origin:__nomatch__" "mamba5:down_blocks.0.*,up_blocks.3.*"; do
        L=${V%%:*}; INC=${V#*:}
        run_one "$L" "$INC" 576 1024 "$BS" "$rep"
        run_one "$L" "$INC" 1024 1792 "$BS" "$rep"
        run_one "$L" "$INC" 1024 1920 "$BS" "$rep"
      done
    done
    echo "  rep $rep done $(date +%H:%M:%S)" | tee -a "$RES_TXT"
  done
fi
echo "BENCH_DONE $(date +%F_%H:%M:%S) mode=$MODE" | tee -a "$RES_TXT"
