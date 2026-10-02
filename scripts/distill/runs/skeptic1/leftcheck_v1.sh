#!/bin/bash
set -u; cd /home/kawa/master_project/StereoCrafter
PY=/home/kawa/miniconda3/envs/stereocrafter/bin/python
S=scripts/distill/runs/skeptic1/leftcheck_v1.py
C=outputs/fulldata_v2/clips; B=outputs/beyond_distil
O=scripts/distill/runs/skeptic1/LEFTCHECK_mp4v.txt
: > $O
student() { case $1 in
  0301) echo $B/0301_smoke1_step800/0301_inpainting_results_sbs.mp4;;
  0204) echo $B/0204_smoke1_step800/0204_inpainting_results_sbs.mp4;;
  0052|0128|0125|0170) echo $B/$1_heldout_smoke1s800/$1_inpainting_results_sbs.mp4;;
  *) echo $B/$1_heldout_s800rest/$1_inpainting_results_sbs.mp4;; esac }
for CL in 0301 0204 0052 0147 0259; do
  $PY $S $CL $C/${CL}_origin/${CL}_inpainting_results_sbs.mp4 \
     s25=$C/${CL}_origin_s25/${CL}_inpainting_results_sbs.mp4 \
     mamba_all8kv2=$C/${CL}_all_8k_v2/${CL}_inpainting_results_sbs.mp4 \
     student=$(student $CL) \
     g125=$C/${CL}_origin_g125/${CL}_inpainting_results_sbs.mp4 2>&1 | grep -v Warning >> $O
done
echo DONE >> $O
