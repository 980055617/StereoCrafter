#!/bin/bash
# 12-clip extension.  usage: chain5b_ext12.sh <ckpt> <label> [waitunit]
# The 8 clips 0042 0125 0128 0141 0170 0225 0251 0259 have NO mamba_ll and NO mamba_s25_ll row anywhere,
# so both baselines have to be produced here before a student row on them means anything.
# PASS A (cheap, 8 steps):  mamba_ll + student  -> answers "does it regress on any of the 12 clips?"
#                           scored immediately, so a usable 12-clip table exists early.
# PASS B (expensive, 25 steps): mamba_s25_ll    -> the headroom denominator, scored afterwards.
set -u
cd /home/kawa/master_project/StereoCrafter
PY=/home/kawa/miniconda3/envs/stereocrafter/bin/python
D=scripts/distill/runs/beyond_distil_mamba
CK=$1; LAB=$2; WAIT=${3:-}
[ -n "$WAIT" ] && { until ! systemctl --user is-active --quiet $WAIT; do sleep 30; done; }
ALL12=0042,0052,0125,0128,0141,0147,0170,0204,0225,0251,0259,0301
REST="0042 0125 0128 0141 0170 0225 0251 0259"

JOB=$D/jobs_ext12a_$LAB.txt; : > $JOB
for CL in $REST; do echo "$CL mamba_ll none" >> $JOB; done
for CL in $REST; do echo "$CL ${LAB}_ll $CK" >> $JOB; done
bash $D/run_driver_v2.sh $JOB 1
CUDA_VISIBLE_DEVICES=1 $PY $D/score_rows_v1.py $D/SCORES_12CLIP_A_$LAB.txt $ALL12 \
  "origin_ll,student_ll,mamba_ll,${LAB}_ll" > $D/score_12clip_a_$LAB.log 2>&1
echo "passA score rc=$?"
SUM_CLIPS=$ALL12 $PY $D/summarize_v1.py $D/TABLE_12CLIP_A_$LAB.txt $D/SCORES_12CLIP_A_$LAB.txt >/dev/null 2>&1
cat $D/TABLE_12CLIP_A_$LAB.txt
echo CHAIN5B_PASSA_DONE $(date +%T)

JOB2=$D/jobs_ext12b_$LAB.txt; : > $JOB2
for CL in $REST; do echo "$CL mamba_s25_ll none" >> $JOB2; done
SK_STEPS=25 bash $D/run_driver_v2.sh $JOB2 1
CUDA_VISIBLE_DEVICES=1 $PY $D/score_rows_v1.py $D/SCORES_12CLIP_B_$LAB.txt $ALL12 \
  "origin_ll,s25_ll,student_ll,mamba_ll,mamba_s25_ll,${LAB}_ll" > $D/score_12clip_b_$LAB.log 2>&1
echo "passB score rc=$?"
SUM_CLIPS=$ALL12 $PY $D/summarize_v1.py $D/TABLE_12CLIP_$LAB.txt $D/SCORES_12CLIP_B_$LAB.txt >/dev/null 2>&1
cat $D/TABLE_12CLIP_$LAB.txt
echo CHAIN5B_DONE $(date +%T)
