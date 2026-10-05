#!/bin/bash
# GPU steps of one scoring stage, run by score_stage_v1.sh under ONE acquisition of /tmp/claude-gpu0.lock (no flock in here;
# never call it without the outer lock).  Same scorers, same arguments, same per-clip processes as before.
# usage: score_gpu_steps_v1.sh <stage_dir> <clip> [<clip> ...]
set -u
cd /home/kawa/master_project/StereoCrafter
L=scripts/distill/runs/deep_20261004/sdedit
PY=/home/kawa/miniconda3/envs/stereocrafter/bin/python
D=$1; shift
echo "GPU_STEPS locked $(date +%F_%T)" | tee -a $D/stage.log
# ---- M1 unregistered LPIPS, one clip per process; ROW lines appended to SCORES.txt
: > $D/SCORES.txt
for c in "$@"; do
  CUDA_VISIBLE_DEVICES=0 SCORE_STEP=4 $PY scripts/distill/runs/fulldata_v2/beyond4/score_clip_ll.py \
    $(grep "^$c=" $D/scorelist.txt | tr '\n' ' ') >> $D/SCORES.txt 2>> $D/SCORES.err
  echo "M1 $c rc=$?" | tee -a $D/stage.log
done
echo "M1 rows=$(grep -c '^ROW' $D/SCORES.txt)" | tee -a $D/stage.log
$PY $L/build_rows_v1.py $D/SCORES.txt $D/ROWS.json | tee -a $D/stage.log
# ---- M1r registered LPIPS, one clip per process
for c in "$@"; do
  if [ "$c" = "0125" ]; then
    mkdir -p $D/reg_wide
    CUDA_VISIBLE_DEVICES=0 SCORE_STEP=4 REG_DDY=-20,20 REG_DDX=-240,40 $PY \
      scripts/distill/runs/more_20261004/eval_robustness/score_registered_v1w.py $D/reg_wide $D/ROWS.json $c > $D/reg_wide/$c.log 2>&1
    echo "M1r-wide $c rc=$?" | tee -a $D/stage.log
  fi
  CUDA_VISIBLE_DEVICES=0 SCORE_STEP=4 $PY scripts/distill/runs/more_20261004/eval_robustness/score_registered_v1.py \
    $D/reg $D/ROWS.json $c > $D/reg/$c.log 2>&1
  echo "M1r $c rc=$?" | tee -a $D/stage.log
done
# ---- M3 temporal, one clip per process (origin_ll is each clip's first row); SKIP_TEMPORAL=1 skips it (smoke only)
if [ "${SKIP_TEMPORAL:-0}" = "1" ]; then
  echo "M3 skipped (SKIP_TEMPORAL=1)" | tee -a $D/stage.log
else
  for c in "$@"; do
    CUDA_VISIBLE_DEVICES=0 $PY scripts/distill/runs/finalcheck_20261004/validate/score_temporal_ll.py \
      $D/temporal/$c.json $(grep "^$c=" $D/scorelist.txt | tr '\n' ' ') > $D/temporal/$c.log 2>&1
    echo "M3 $c rc=$?" | tee -a $D/stage.log
  done
  $PY -c "import json,sys; d={}; [d.update(json.load(open(f'$D/temporal/{c}.json'))) for c in sys.argv[1:]]; json.dump(d, open('$D/temporal/temporal.json','w'), indent=1)" "$@"
  echo "M3 merged rc=$?" | tee -a $D/stage.log
fi
echo "GPU_STEPS done $(date +%F_%T)" | tee -a $D/stage.log
