#!/bin/bash
# final_judge: reproduce input_side's P1 rows (R1 vs RF, origin and deliverable, 4 clips) with THAT lane's own scorer
# (score_input_v1.py, GT registered to each input variant's own warp), into a NEW dir of this lane.  GPU 0 lock per job.
set -u
cd /home/kawa/master_project/StereoCrafter
set +u; source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate stereocrafter; set -u
SC=scripts/distill/runs/deep_20261004/input_side/score_input_v1.py
INROOT=/mnt/ssd_data/deep_20261004/input_side/inputs_v1
SD=outputs/deep_20261004/final_judge/inputside_repro_v1
LOG=scripts/distill/runs/deep_20261004/final_judge/inputside_repro_v1.log
mkdir -p $SD
for C in 0301 0204 0052 0147; do for I in R1 RF; do
  if [ -e "$SD/${C}__${I}.json" ]; then echo "SKIP $C $I" >> $LOG; continue; fi
  CUDA_VISIBLE_DEVICES=0 SCORE_STEP=4 flock /tmp/claude-gpu0.lock python $SC $SD $C $INROOT/$C/$I \
    origin=outputs/deep_20261004/input_side/clips/${C}_${I}_origin/${C}_inpainting_results_sbs.mkv \
    deliv=outputs/deep_20261004/input_side/clips/${C}_${I}_deliv/${C}_inpainting_results_sbs.mkv > $SD/${C}__${I}.log 2>&1
  echo "SCORE $C $I rc=$? $(date +%T)" >> $LOG
done; done
echo "DONE $(date +%T)" >> $LOG
