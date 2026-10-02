#!/bin/bash
# GPU-1 lane, run entirely in GPU-0's shadow.  Renders, at the DEPLOYED config through the LOSSLESS
# FFV1 path, on the four DEV-split clips (0040 0091 0184 0245 -- fulldata_v1's designated
# "model selection / stop rule" split, gt=True, never trained by any run):
#     1. the shipped 5-slot Mamba baseline
#     2. mstudent1 step600 (its selected ckpt) and step800 (matched optimiser budget)
#     3. every mstudent2 rung as training writes it
# Checkpoint selection is by SAMPLED LPIPS on these dev clips -- NOT on the test clips mstudent1 was
# selected on, and NOT on training loss (the project's loss/quality anti-correlation, 5 sightings).
set -u
cd /home/kawa/master_project/StereoCrafter
PY=/home/kawa/miniconda3/envs/stereocrafter/bin/python
D=scripts/distill/runs/beyond_distil_mamba_scaled
M1=scripts/distill/runs/beyond_distil_mamba/mstudent1
DEV="0040 0091 0184 0245"
TD=$D/mstudent2
LOG=$D/chainB_progress.txt

echo "waiting for pass B (bdm-chain5c) to release GPU 1 ... $(date +%T)" | tee -a $LOG
until ! systemctl --user is-active --quiet bdm-chain5c; do sleep 30; done
echo "GPU 1 free $(date +%T)" | tee -a $LOG

# ---- 1-2: the dev baselines (shipped Mamba) and the mstudent1 reference rungs
J=$D/jobs_dev_base.txt; : > $J
for C in $DEV; do echo "$C mamba_ll none" >> $J; done
for C in $DEV; do echo "$C mstudent1_step600_ll $M1/step600.pt" >> $J; done
for C in $DEV; do echo "$C mstudent1_step800_ll $M1/step800.pt" >> $J; done
bash $D/run_driver_v3.sh $J 1
echo "DEV_BASELINES_DONE $(date +%T)" | tee -a $LOG

# ---- 3: every mstudent2 rung, as soon as the trainer has written it
for S in 200 400 800 1200 1600 2000; do
  echo "waiting for $TD/step$S.pt ... $(date +%T)" | tee -a $LOG
  while true; do
    grep -q "^saved step$S.pt$" $D/train_mstudent2.log 2>/dev/null && break
    systemctl --user is-active --quiet bdms-chainA || { echo "TRAINER GONE before step$S" | tee -a $LOG; break 2; }
    sleep 30
  done
  [ -f $TD/step$S.pt ] || { echo "MISSING $TD/step$S.pt" | tee -a $LOG; continue; }
  J=$D/jobs_dev_s$S.txt; : > $J
  for C in $DEV; do echo "$C mstudent2_step${S}_ll $TD/step$S.pt" >> $J; done
  bash $D/run_driver_v3.sh $J 1
  echo "RUNG_${S}_DONE $(date +%T)" | tee -a $LOG
done
echo CHAINB_DONE $(date +%T) | tee -a $LOG
