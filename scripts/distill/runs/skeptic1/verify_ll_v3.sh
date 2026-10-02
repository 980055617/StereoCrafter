#!/bin/bash
# reproduce beyond4's own verify_lossless.py CLAIM1/2/3 on TWO clips they did not print in FAITHFULNESS.txt
set -u; cd /home/kawa/master_project/StereoCrafter
PY=/home/kawa/miniconda3/envs/stereocrafter/bin/python
O=scripts/distill/runs/skeptic1/VERIFY_LOSSLESS_v3.txt; : > $O
for CL in 0204 0147; do
  echo "--- $CL ---" >> $O
  $PY scripts/distill/runs/fulldata_v2/beyond4/verify_lossless.py \
    outputs/beyond4_lossless/clips/${CL}_origin_ll/${CL}_inpainting_results_sbs.mkv \
    outputs/fulldata_v2/clips/${CL}_origin/${CL}_inpainting_results_sbs.mp4 \
    video_data/splatting/${CL}_splatting_results.mp4 2>&1 | grep -viE "warning|^ *@torch" >> $O
done
echo DONE >> $O
