#!/bin/bash
# more_20261004 / lane mamba_scope, driver v1.  See PREREG.txt in this directory.
# Pass A = scripts/distill/bench2.py UNMODIFIED (totals); pass B = bench_slots16_v1.py (per-slot, all 16 spatial attn1).
# BS=1 everywhere; Mamba = the deliverable's knobs (gated_residual, GATE 1.0, DS 128, EXP 1, fwd); no checkpoint.
#
# usage (must be launched INSIDE `flock /tmp/claude-gpu0.lock`; the caller holds the lock for the whole run):
#   run_v1.sh OUTDIR smoke|full
#     smoke : pass B mamba10 @1024x1792 (rep 0) + pass A mamba10 @576x1024 (rep 0); never counted
#     full  : pass A reps 1..3 x config (origin mamba5 mamba10 all16) x res (576x1024 1024x1792 1024x1920)
#             then pass B reps 1..2 x config (origin mamba10 all16) x res
set -u
OUTDIR=$1; MODE=$2
REPO=/home/kawa/master_project/StereoCrafter
LANE=$REPO/scripts/distill/runs/more_20261004/mamba_scope
cd "$REPO"
set +u; source ~/miniconda3/etc/profile.d/conda.sh; conda activate stereocrafter; set -u
export CUDA_VISIBLE_DEVICES=0
unset PYTORCH_CUDA_ALLOC_CONF
for v in $(env | grep -oE '^MAMBA_[A-Za-z0-9_]+'); do unset "$v"; done

mkdir -p "$OUTDIR/logs" "$OUTDIR/profiles"
RES_TXT="$OUTDIR/bench_results.txt"      # RESULT lines (verbatim) + RUNMETA lines
CHK="$OUTDIR/exclusivity_checks.txt"     # nvidia-smi snapshots before/after every process
GPULOG="$OUTDIR/gpu_monitor.csv"         # 2 s sampler over the whole run, both GPUs
env | sort > "$OUTDIR/env_at_start.txt"
md5sum scripts/distill/bench2.py "$LANE/bench_slots16_v1.py" scripts/distill/runs/slotbudget/bench_slots_v2.py \
  "$LANE/run_v1.sh" > "$OUTDIR/script_md5.txt"
diff scripts/distill/runs/slotbudget/bench_slots_v2.py "$LANE/bench_slots16_v1.py" > "$OUTDIR/passB_copy_diff.txt"
python3 -c "import torch, diffusers, sys; print('python', sys.version.split()[0], 'torch', torch.__version__, 'cuda', torch.version.cuda, 'diffusers', diffusers.__version__, 'gpu', torch.cuda.get_device_name(0))" > "$OUTDIR/versions.txt" 2>&1

nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,temperature.gpu,clocks.sm,power.draw \
  --format=csv,noheader -l 2 > "$GPULOG" 2>&1 &
MON=$!
trap 'kill $MON 2>/dev/null' EXIT

echo "RUN_START $(date +%F_%H:%M:%S) mode=$MODE outdir=$OUTDIR" | tee -a "$RES_TXT"

snap() {  # tag phase
  {
    echo "$2 $1 $(date +%F_%H:%M:%S)"
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,temperature.gpu,clocks.sm --format=csv,noheader
    echo "compute-apps:"
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
  } >> "$CHK" 2>&1
}

attempt() {  # pass L INC H W tag -> 0 iff rc=0 and a RESULT line exists
  local P=$1 L=$2 INC=$3 H=$4 W=$5 tag=$6
  local log="$OUTDIR/logs/${P}_${tag}.log"
  snap "$P/$tag" PRE
  local t0; t0=$(date +%s.%N)
  if [ "$P" = "A" ]; then
    H=$H W=$W BS=1 GATE=1.0 DS=128 EXP=1 BIDIR=fwd \
      python3 scripts/distill/bench2.py "$tag" "$INC" "__nomatch__" > "$log" 2>&1
  else
    H=$H W=$W BS=1 GATE=1.0 DS=128 EXP=1 BIDIR=fwd SB_JSON="$OUTDIR/profiles/${tag}.json" \
      python3 "$LANE/bench_slots16_v1.py" "$tag" "$INC" "__nomatch__" > "$log" 2>&1
  fi
  local rc=$?
  local t1; t1=$(date +%s.%N)
  snap "$P/$tag" POST
  local res; res=$(grep -E '^RESULT' "$log" | tail -1)
  local repl; repl=$(grep -oE 'total_replaced=[0-9]+' "$log" | tail -1)
  [ -n "$res" ] && echo "PASS$P $res" | tee -a "$RES_TXT"
  echo "RUNMETA pass=$P tag=$tag config=$L include=$INC H=$H W=$W BS=1 rc=$rc ${repl:-total_replaced=NA} start=$t0 end=$t1 has_result=$([ -n "$res" ] && echo 1 || echo 0)" | tee -a "$RES_TXT"
  [ $rc -eq 0 ] && [ -n "$res" ]
}

run_one() {  # pass L INC H W rep : one process, retried at most once
  local P=$1 L=$2 INC=$3 H=$4 W=$5 rep=$6
  local tag="${L}_bs1_h${H}w${W}_r${rep}"
  if ! attempt "$P" "$L" "$INC" "$H" "$W" "$tag"; then
    echo "FAILED pass=$P $tag -> retry once; last lines of the log:" | tee -a "$RES_TXT"
    tail -5 "$OUTDIR/logs/${P}_${tag}.log" | sed 's/^/    /' | tee -a "$RES_TXT"
    attempt "$P" "$L" "$INC" "$H" "$W" "${tag}_retry" || echo "FAILED_TWICE pass=$P $tag" | tee -a "$RES_TXT"
  fi
}

CFG_origin="__nomatch__"
CFG_mamba5="down_blocks.0.*,up_blocks.3.*"
CFG_mamba10="down_blocks.0.*,up_blocks.3.*,down_blocks.1.*,up_blocks.2.*"
CFG_all16="*"
cfg() { local v="CFG_$1"; echo "${!v}"; }

if [ "$MODE" = "smoke" ]; then
  run_one B mamba10 "$(cfg mamba10)" 1024 1792 0
  run_one A mamba10 "$(cfg mamba10)" 576 1024 0
else
  for rep in 1 2 3; do
    for L in origin mamba5 mamba10 all16; do
      run_one A "$L" "$(cfg $L)" 576 1024 "$rep"
      run_one A "$L" "$(cfg $L)" 1024 1792 "$rep"
      run_one A "$L" "$(cfg $L)" 1024 1920 "$rep"
    done
    echo "  passA rep $rep done $(date +%H:%M:%S)" | tee -a "$RES_TXT"
  done
  for rep in 1 2; do
    for L in origin mamba10 all16; do
      run_one B "$L" "$(cfg $L)" 576 1024 "$rep"
      run_one B "$L" "$(cfg $L)" 1024 1792 "$rep"
      run_one B "$L" "$(cfg $L)" 1024 1920 "$rep"
    done
    echo "  passB rep $rep done $(date +%H:%M:%S)" | tee -a "$RES_TXT"
  done
fi
echo "RUN_DONE $(date +%F_%H:%M:%S) mode=$MODE" | tee -a "$RES_TXT"
