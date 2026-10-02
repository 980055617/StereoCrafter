#!/bin/bash
# SLOT BUDGET phase 2 (after the profile): deployed-path module-profile cross-check, then the
# 2-slot-alone quality lane (8 missing clips) and the 12-clip lossless scoring.
set -u
cd /home/kawa/master_project/StereoCrafter
R=scripts/distill/runs/slotbudget
bash $R/deployed_modprofile_v1.sh
bash $R/qual_driver_v1.sh $PWD/$R/jobs_down0_missing8.txt 0
bash $R/qual_score_v1.sh
echo "SLOTBUDGET_PHASE2_DONE $(date +%F_%H:%M:%S)"
