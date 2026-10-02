#!/bin/bash
# Does the distilled student make the same artefact/detail trade as s25 and g125?
# Same GT-defined-region metric, same lossless videos, four regime-spanning clips.
set -u; cd /home/kawa/master_project/StereoCrafter
PY=/home/kawa/miniconda3/envs/stereocrafter/bin/python
R=scripts/distill/runs/fulldata_v2/beyond4/ringing_metrics.py
B4=outputs/beyond4_lossless/clips; SK=outputs/skeptic1_stack/clips
O=scripts/distill/runs/skeptic1/RINGING_STUDENT.txt; : > $O
export SCORE_STEP=8
run(){ CL=$1; DY=$2; DX=$3
  echo "=== $CL (offset $DY,$DX) ===" >> $O
  $PY $R $CL $DY $DX \
    origin=$B4/${CL}_origin_ll/${CL}_inpainting_results_sbs.mkv \
    s25=$B4/${CL}_s25_ll/${CL}_inpainting_results_sbs.mkv \
    g125=$B4/${CL}_g125_ll/${CL}_inpainting_results_sbs.mkv \
    student=$SK/${CL}_student_ll/${CL}_inpainting_results_sbs.mkv \
    mamba=$SK/${CL}_mamba_ll/${CL}_inpainting_results_sbs.mkv \
    mamba_s25=$SK/${CL}_mamba_s25_ll/${CL}_inpainting_results_sbs.mkv 2>&1 | grep -v Warning >> $O
}
until ! systemctl --user is-active --quiet sk-gpu0 && ! systemctl --user is-active --quiet sk-gpu1; do sleep 20; done
run 0301 -28 0
run 0204 -28 0
run 0052 -12 -12
run 0147 -12 -12
echo DONE >> $O
