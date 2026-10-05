#!/bin/bash
# M3 (temporal) GPU step of a stage, split out of score_gpu_steps_v1.sh for the S2 flow; run ONLY under the GPU-0 lock
# (no flock inside).  usage: score_gpu_temporal_v1.sh <stage_dir> <clip> [<clip> ...]
set -u
cd /home/kawa/master_project/StereoCrafter
PY=/home/kawa/miniconda3/envs/stereocrafter/bin/python
D=$1; shift
if flock -n /tmp/claude-gpu0.lock true; then echo "NOT UNDER /tmp/claude-gpu0.lock -> abort"; exit 7; fi
echo "GPU_TEMPORAL locked $(date +%F_%T)" | tee -a $D/stage.log
# ---- M3 temporal, one clip per process (origin_ll is each clip's first row); SKIP_TEMPORAL=1 skips it (smoke only)
  for c in "$@"; do
    CUDA_VISIBLE_DEVICES=0 $PY scripts/distill/runs/finalcheck_20261004/validate/score_temporal_ll.py \
      $D/temporal/$c.json $(grep "^$c=" $D/scorelist.txt | tr '\n' ' ') > $D/temporal/$c.log 2>&1
    echo "M3 $c rc=$?" | tee -a $D/stage.log
  done
  $PY -c "import json,sys; d={}; [d.update(json.load(open(f'$D/temporal/{c}.json'))) for c in sys.argv[1:]]; json.dump(d, open('$D/temporal/temporal.json','w'), indent=1)" "$@"
  echo "M3 merged rc=$?" | tee -a $D/stage.log
echo "GPU_TEMPORAL done $(date +%F_%T)" | tee -a $D/stage.log
