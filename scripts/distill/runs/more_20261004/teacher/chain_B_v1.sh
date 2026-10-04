#!/bin/bash
# chain B: Stage 1a (PREREG S1) -- H13 D16 D25 C25 full renders on 0301 -> integrity checks -> scoring.
set -u
cd /home/kawa/master_project/StereoCrafter
D=scripts/distill/runs/more_20261004/teacher
O=outputs/more_20261004/teacher/clips
PY=/home/kawa/miniconda3/envs/stereocrafter/bin/python
echo "CHAIN_B_START $(date +%F_%T)"
bash $D/run_driver_teacher_v1.sh $D/jobs_10_stage1a.txt
CUDA_VISIBLE_DEVICES="" $PY $D/check_fingerprints_v1.py $D/FINGERPRINTS_stage1a.txt $O/0301_gate_euler25 $O/0301_H13 $O/0301_D16 $O/0301_D25 $O/0301_C25
CUDA_VISIBLE_DEVICES="" $PY $D/check_passthrough_v1.py $D/PASSTHROUGH_stage1a.txt $D/passthrough_ref_cache.json $O/0301_gate_euler25 $O/0301_H13 $O/0301_D16 $O/0301_D25 $O/0301_C25
bash $D/score_v1.sh $D/SCORES_stage1a.txt $D/scorelist_stage1a.txt
grep -E "^ROW" $D/SCORES_stage1a.txt | sed -E 's/.*tag=([^ ]+).*lpips=([^ ]+) sharp=([^ ]+) gtSharp=([^ ]+).*/\1 lpips=\2 sharp=\3 gt=\4/'
echo "CHAIN_B_DONE $(date +%F_%T)"
