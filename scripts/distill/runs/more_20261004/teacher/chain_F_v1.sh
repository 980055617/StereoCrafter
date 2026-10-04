#!/bin/bash
# chain F: PREREG_ADDENDUM_3 post-hoc C25 on 0147 + 0052 -> cross-clip fingerprint consistency + pass-through -> score.
set -u
cd /home/kawa/master_project/StereoCrafter
D=scripts/distill/runs/more_20261004/teacher
O=outputs/more_20261004/teacher/clips
PY=/home/kawa/miniconda3/envs/stereocrafter/bin/python
echo "CHAIN_F_START $(date +%F_%T)"
bash $D/run_driver_teacher_v1.sh $D/jobs_30_posthoc_C25.txt
CUDA_VISIBLE_DEVICES="" $PY $D/check_passthrough_v1.py $D/PASSTHROUGH_posthoc.txt $D/passthrough_ref_cache_posthoc.json $O/0147_C25 $O/0052_C25
bash $D/score_v1.sh $D/SCORES_posthoc_C25.txt $D/scorelist_posthoc_C25.txt
grep -E "^ROW" $D/SCORES_posthoc_C25.txt
echo "CHAIN_F_DONE $(date +%F_%T)"
