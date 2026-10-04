#!/bin/bash
# finalcheck_20261004 / bench lane, DIAGNOSTIC dtype arm (PREREG_ADDENDUM_dtype.txt; runs only if X1 failed).
# Same protocol/env/order as bench_v1.sh, but the timed script is bench2_dtype_diag.py (one-line copy of
# scripts/distill/bench2.py: dtype from env DT).  Per rep: fp16 copy-identity control (origin, mamba5 at BS=1, 576x1024),
# then DT=bf16 for all 12 configurations.
# usage: bench_dtype_v1.sh OUTDIR
set -u
cd /home/kawa/master_project/StereoCrafter
OUTDIR=$1
D=scripts/distill
S=$D/runs/finalcheck_20261004/bench/bench2_dtype_diag.py
mkdir -p "$OUTDIR/logs"
RES_TXT="$OUTDIR/bench_results.txt"
CHK="$OUTDIR/exclusivity_checks.txt"
GPULOG="$OUTDIR/gpu_monitor.csv"

export CUDA_VISIBLE_DEVICES=0
unset PYTORCH_CUDA_ALLOC_CONF
for v in $(env | grep -oE '^MAMBA_[A-Za-z0-9_]+'); do unset "$v"; done
env | sort > "$OUTDIR/env_at_start.txt"
md5sum $D/bench2.py $S > "$OUTDIR/bench2_md5.txt"
diff $D/bench2.py $S > "$OUTDIR/diag_copy_diff.txt"
python3 -c "import torch, diffusers, sys; print('python', sys.version.split()[0], 'torch', torch.__version__, 'cuda', torch.version.cuda, 'diffusers', diffusers.__version__, 'gpu', torch.cuda.get_device_name(0))" > "$OUTDIR/versions.txt" 2>&1

nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,temperature.gpu,clocks.sm,power.draw \
  --format=csv,noheader -l 2 > "$GPULOG" 2>&1 &
MON=$!
trap 'kill $MON 2>/dev/null' EXIT

echo "BENCH_START $(date +%F_%H:%M:%S) mode=dtype outdir=$OUTDIR" | tee -a "$RES_TXT"
nvidia-smi --query-gpu=index,name,memory.used,temperature.gpu --format=csv,noheader | tee -a "$RES_TXT"

snap() {
  {
    echo "$2 $1 $(date +%F_%H:%M:%S)"
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,temperature.gpu,clocks.sm --format=csv,noheader
    echo "compute-apps:"
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
  } >> "$CHK" 2>&1
}

attempt() {  # L INC H W BS DT tag
  local L=$1 INC=$2 H=$3 W=$4 BS=$5 DTV=$6 tag=$7
  local log="$OUTDIR/logs/bench_${tag}.log"
  snap "$tag" PRE
  local t0; t0=$(date +%s.%N)
  DT=$DTV H=$H W=$W BS=$BS GATE=1.0 DS=128 EXP=1 BIDIR=fwd \
    python3 $S "$tag" "$INC" "__nomatch__" > "$log" 2>&1
  local rc=$?
  local t1; t1=$(date +%s.%N)
  snap "$tag" POST
  local res; res=$(grep -E '^RESULT' "$log" | tail -1)
  local repl; repl=$(grep -oE 'total_replaced=[0-9]+' "$log" | tail -1)
  [ -n "$res" ] && echo "$res" | tee -a "$RES_TXT"
  echo "RUNMETA tag=$tag config=$L include=$INC H=$H W=$W BS=$BS DT=$DTV rc=$rc ${repl:-total_replaced=NA} start=$t0 end=$t1 has_result=$([ -n "$res" ] && echo 1 || echo 0)" | tee -a "$RES_TXT"
  [ $rc -eq 0 ] && [ -n "$res" ]
}

run_one() {  # L INC H W BS DT rep
  local L=$1 INC=$2 H=$3 W=$4 BS=$5 DTV=$6 rep=$7
  local tag="${L}_${DTV}_bs${BS}_h${H}w${W}_r${rep}"
  if ! attempt "$L" "$INC" "$H" "$W" "$BS" "$DTV" "$tag"; then
    echo "FAILED $tag -> retry once; last lines of the log:" | tee -a "$RES_TXT"
    tail -5 "$OUTDIR/logs/bench_${tag}.log" | sed 's/^/    /' | tee -a "$RES_TXT"
    attempt "$L" "$INC" "$H" "$W" "$BS" "$DTV" "${tag}_retry" || echo "FAILED_TWICE $tag" | tee -a "$RES_TXT"
  fi
}

for rep in 1 2 3; do
  run_one origin "__nomatch__" 576 1024 1 fp16 "$rep"
  run_one mamba5 "down_blocks.0.*,up_blocks.3.*" 576 1024 1 fp16 "$rep"
  for BS in 2 1; do
    for V in "origin:__nomatch__" "mamba5:down_blocks.0.*,up_blocks.3.*"; do
      L=${V%%:*}; INC=${V#*:}
      run_one "$L" "$INC" 576 1024 "$BS" bf16 "$rep"
      run_one "$L" "$INC" 1024 1792 "$BS" bf16 "$rep"
      run_one "$L" "$INC" 1024 1920 "$BS" bf16 "$rep"
    done
  done
  echo "  rep $rep done $(date +%H:%M:%S)" | tee -a "$RES_TXT"
done
echo "BENCH_DONE $(date +%F_%H:%M:%S) mode=dtype" | tee -a "$RES_TXT"
