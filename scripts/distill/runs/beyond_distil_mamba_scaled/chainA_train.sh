#!/bin/bash
# SCALED Mamba-side step distillation (option A), round 2.  Same objective / sub-grid / step subset
# {4,5,6} / MEASURED gains {0.7177,0.8926,0.9905} / x0-space residual / AdamW lr 1e-5 / clip 1.0 /
# M=4 Karras substeps / NO on-policy refresh as mstudent1.  THREE things change:
#   1. TRAINING SET: 10 clips from fulldata_v1's TRAIN split (curve_13 minus 0160 -- the splits file's
#      own "contaminated continuity reference" -- and minus the two long clips 0358/0335 whose 151
#      extra windows would not fit the GPU budget).  134 windows / 402 target points, vs mstudent1's
#      2 TEST clips (0301,0204) / 26 windows / 78 points.  NONE of the 12 test clips and NONE of the
#      8 dev clips appear here, so every number in the headline table is fully held out.
#   2. 2000 optimiser steps with a dense checkpoint ladder, so step800 answers "more clips at MATCHED
#      optimiser budget" and the later rungs answer "more clips AND more steps".
#   3. Checkpoints are selected on the DEV split (0040,0091,0184,0245), which fulldata_v1's rules
#      designate for "model selection / stop rule" -- mstudent1 selected on test clips.
set -u
cd /home/kawa/master_project/StereoCrafter
PY=/home/kawa/miniconda3/envs/stereocrafter/bin/python
D=scripts/distill/runs/beyond_distil_mamba_scaled
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BD_CLIPS=0094,0011,0177,0144,0179,0105,0286,0078,0165,0114
export BD_STEP_SUBSET=4,5,6 BD_M=4 BD_TRAIN_STEPS=2000 BD_LR=1e-5
export BD_SAVE=200,400,800,1200,1600,2000
export BD_WEIGHTS=4:0.7177,5:0.8926,6:0.9905
export BD_SLOTS=up3
$PY $D/train_beyond_mamba_scaled.py mstudent2 > $D/train_mstudent2.log 2>&1
echo "TRAIN rc=$? $(date +%T)"
grep -E "^OUT|BD_TRAIN_DONE|\[post\]|DONE " $D/train_mstudent2.log | tail -20
echo CHAINA_DONE
