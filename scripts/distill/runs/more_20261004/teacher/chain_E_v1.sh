#!/bin/bash
# chain E: conditional C25b (S_churn=10) on 0301 -> fingerprint + pass-through checks -> score.
set -u
cd /home/kawa/master_project/StereoCrafter
D=scripts/distill/runs/more_20261004/teacher
O=outputs/more_20261004/teacher/clips
PY=/home/kawa/miniconda3/envs/stereocrafter/bin/python
echo "CHAIN_E_START $(date +%F_%T)"
bash $D/run_driver_teacher_v1.sh $D/jobs_11_C25b.txt
CUDA_VISIBLE_DEVICES="" $PY $D/check_fingerprints_v1.py $D/FINGERPRINTS_C25b.txt $O/0301_gate_euler25 $O/0301_C25b
CUDA_VISIBLE_DEVICES="" $PY $D/check_passthrough_v1.py $D/PASSTHROUGH_C25b.txt $D/passthrough_ref_cache.json $O/0301_C25b
bash $D/score_v1.sh $D/SCORES_C25b.txt $D/scorelist_C25b.txt
grep -E "^ROW" $D/SCORES_C25b.txt
echo "CHAIN_E_DONE $(date +%F_%T)"
