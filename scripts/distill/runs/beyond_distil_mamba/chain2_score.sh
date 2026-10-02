#!/bin/bash
# Score EVERY Mamba-side checkpoint at the DEPLOYED config through the LOSSLESS path, on the two trained
# clips (0301, 0204) and two held-out clips (0052, 0147).  Checkpoint selection is by SAMPLED LPIPS only --
# the project's loss/quality anti-correlation has now been observed four times.
set -u
cd /home/kawa/master_project/StereoCrafter
PY=/home/kawa/miniconda3/envs/stereocrafter/bin/python
D=scripts/distill/runs/beyond_distil_mamba
until ! systemctl --user is-active --quiet bdm-chain1; do sleep 30; done
TRAINDIR=$(grep -m1 '^OUT ' $D/train_mstudent1.log | awk '{print $2}')
echo "TRAINDIR=$TRAINDIR"
[ -d "$TRAINDIR" ] || { echo "no train dir"; exit 3; }
TAG=$(basename $TRAINDIR)
JOB=$D/jobs_score_$TAG.txt
: > $JOB
for CL in 0301 0204 0052 0147; do
  for S in 100 200 400 600 800; do
    [ -f "$TRAINDIR/step$S.pt" ] && echo "$CL ${TAG}_step${S}_ll $TRAINDIR/step$S.pt" >> $JOB
  done
done
wc -l $JOB
bash $D/run_driver_v1.sh $JOB 1
LAB="origin_ll,s25_ll,student_ll,mamba_ll,mamba_s25_ll,mamba_oracle456_ll,mamba_oracleALL_ll"
for S in 100 200 400 600 800; do
  [ -f "$TRAINDIR/step$S.pt" ] && LAB="$LAB,${TAG}_step${S}_ll"
done
CUDA_VISIBLE_DEVICES=1 $PY $D/score_rows_v1.py $D/SCORES_4CLIP_$TAG.txt 0301,0204,0052,0147 "$LAB" \
  > $D/score_4clip_$TAG.log 2>&1
echo "score rc=$?"
SUM_CLIPS=0301,0204,0052,0147 $PY $D/summarize_v1.py $D/TABLE_4CLIP_$TAG.txt $D/SCORES_4CLIP_$TAG.txt \
  > /dev/null 2>&1
cat $D/TABLE_4CLIP_$TAG.txt
echo CHAIN2_DONE
