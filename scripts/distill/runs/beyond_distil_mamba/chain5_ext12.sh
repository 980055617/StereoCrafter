#!/bin/bash
# 12-clip extension, only run if the 4-clip set PASSES.  usage: chain5_ext12.sh <ckpt> <label> [waitunit]
# The 8 clips 0042 0125 0128 0141 0170 0225 0251 0259 have NO mamba_ll and NO mamba_s25_ll row anywhere,
# so those two baselines have to be produced here before the student row means anything.
set -u
cd /home/kawa/master_project/StereoCrafter
PY=/home/kawa/miniconda3/envs/stereocrafter/bin/python
D=scripts/distill/runs/beyond_distil_mamba
CK=$1; LAB=$2; WAIT=${3:-}
[ -n "$WAIT" ] && { until ! systemctl --user is-active --quiet $WAIT; do sleep 30; done; }
REST="0042 0125 0128 0141 0170 0225 0251 0259"
JOB=$D/jobs_ext12_$LAB.txt
: > $JOB
for CL in $REST; do echo "$CL mamba_ll none" >> $JOB; done
for CL in $REST; do echo "$CL ${LAB}_ll $CK" >> $JOB; done
bash $D/run_driver_v2.sh $JOB 1
# mamba+s25 for the same 8 clips (25 sampler steps; same driver, SK_STEPS overridden)
JOB2=$D/jobs_ext12_s25_$LAB.txt
: > $JOB2
for CL in $REST; do echo "$CL mamba_s25_ll none" >> $JOB2; done
SK_STEPS=25 bash $D/run_driver_v2.sh $JOB2 1
LABS="origin_ll,s25_ll,student_ll,mamba_ll,mamba_s25_ll,${LAB}_ll"
CUDA_VISIBLE_DEVICES=1 $PY $D/score_rows_v1.py $D/SCORES_12CLIP_$LAB.txt \
  0042,0052,0125,0128,0141,0147,0170,0204,0225,0251,0259,0301 "$LABS" > $D/score_12clip_$LAB.log 2>&1
echo "score rc=$?"
SUM_CLIPS=0042,0052,0125,0128,0141,0147,0170,0204,0225,0251,0259,0301 \
  $PY $D/summarize_v1.py $D/TABLE_12CLIP_$LAB.txt $D/SCORES_12CLIP_$LAB.txt > /dev/null 2>&1
cat $D/TABLE_12CLIP_$LAB.txt
echo CHAIN5_DONE
