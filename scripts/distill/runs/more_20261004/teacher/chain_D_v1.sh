#!/bin/bash
# chain D: X1 Euler s50 (paired) on 0301 -> fingerprint + pass-through checks -> score.
set -u
cd /home/kawa/master_project/StereoCrafter
D=scripts/distill/runs/more_20261004/teacher
O=outputs/more_20261004/teacher/clips
PY=/home/kawa/miniconda3/envs/stereocrafter/bin/python
echo "CHAIN_D_START $(date +%F_%T)"
bash $D/run_driver_teacher_v2.sh $D/jobs_21_X1_s50.txt
CUDA_VISIBLE_DEVICES="" $PY $D/check_fingerprints_v1.py $D/FINGERPRINTS_X1.txt $O/0301_gate_euler25 $O/0301_X1s50
CUDA_VISIBLE_DEVICES="" $PY $D/check_passthrough_v1.py $D/PASSTHROUGH_X1.txt $D/passthrough_ref_cache.json $O/0301_X1s50
bash $D/score_v1.sh $D/SCORES_X1_s50.txt $D/scorelist_X1_s50.txt
grep -E "^ROW" $D/SCORES_X1_s50.txt
echo "CHAIN_D_DONE $(date +%F_%T)"
