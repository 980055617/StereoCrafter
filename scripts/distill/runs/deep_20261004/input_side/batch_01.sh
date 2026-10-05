#!/bin/bash
# input_side batch 1 (~34 min).  Run ONLY as:  flock /tmp/claude-gpu0.lock bash batch_01.sh
set -u
cd /home/kawa/master_project/StereoCrafter
D=scripts/distill/runs/deep_20261004/input_side
PY=/home/kawa/miniconda3/envs/stereocrafter/bin/python
if flock -n /tmp/claude-gpu0.lock true; then echo "NOT UNDER /tmp/claude-gpu0.lock -> abort"; exit 7; fi
echo "BATCH1_LOCKED $(date +%F_%T)"
bash $D/run_driver_input_nolock_v1.sh $D/jobs_b1_renders.txt 0
for c in 0052 0147; do
  CUDA_VISIBLE_DEVICES=0 $PY $D/fit_scale_v1.py $c /mnt/ssd_data/deep_20261004/input_side/fit_v1 > /mnt/ssd_data/deep_20261004/input_side/logs/fit_v1_$c.log 2>&1
  echo "FIT $c rc=$? $(date +%T)"
done
bash $D/run_score_nolock_v1.sh $D/scorelist_b1_g2.txt outputs/deep_20261004/input_side/scores_v1
bash $D/run_score_nolock_v1.sh $D/scorelist_b1_inputonly.txt outputs/deep_20261004/input_side/scores_inputonly_v1
bash $D/run_score_nolock_v1.sh $D/scorelist_02_probe1_0301.txt outputs/deep_20261004/input_side/scores_v1
echo "BATCH1_DONE $(date +%F_%T)"
