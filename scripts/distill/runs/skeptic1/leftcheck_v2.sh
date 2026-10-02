#!/bin/bash
# left-half bleed: (a) remaining mp4v clips, (b) the LOSSLESS outputs (expect bit-identical)
set -u; cd /home/kawa/master_project/StereoCrafter
PY=/home/kawa/miniconda3/envs/stereocrafter/bin/python
S=scripts/distill/runs/skeptic1/leftcheck_v1.py
C=outputs/fulldata_v2/clips; B=outputs/beyond_distil; L=outputs/beyond4_lossless/clips
O=scripts/distill/runs/skeptic1/LEFTCHECK_v2.txt; : > $O
echo "### mp4v era (remaining clips)" >> $O
$PY $S 0147 $C/0147_origin/0147_inpainting_results_sbs.mp4 \
   s25=$C/0147_origin_s25/0147_inpainting_results_sbs.mp4 \
   mamba_all8kv2=$C/0147_all_8k_v2/0147_inpainting_results_sbs.mp4 \
   student=$B/0147_heldout_s800rest/0147_inpainting_results_sbs.mp4 2>&1 | grep -v Warning >> $O
$PY $S 0259 $C/0259_origin/0259_inpainting_results_sbs.mp4 \
   s25=$C/0259_origin_s25/0259_inpainting_results_sbs.mp4 \
   mamba_all8kv2=$C/0259_all_8k_v2/0259_inpainting_results_sbs.mp4 \
   student=$B/0259_heldout_s800rest/0259_inpainting_results_sbs.mp4 2>&1 | grep -v Warning >> $O
echo "### LOSSLESS era (beyond4) -- expect LEFT_BIT_IDENTICAL" >> $O
for CL in 0301 0052 0147 0204; do
  $PY $S $CL $L/${CL}_origin_ll/${CL}_inpainting_results_sbs.mkv \
     s25_ll=$L/${CL}_s25_ll/${CL}_inpainting_results_sbs.mkv \
     g125_ll=$L/${CL}_g125_ll/${CL}_inpainting_results_sbs.mkv 2>&1 | grep -v Warning >> $O
done
echo DONE >> $O
