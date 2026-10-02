#!/bin/bash
set -u; cd /home/kawa/master_project/StereoCrafter
PY=/home/kawa/miniconda3/envs/stereocrafter/bin/python
O=scripts/distill/runs/skeptic1/VERIFY_LOSSLESS_v2.txt; : > $O
for CL in 0301 0052 0147; do
  M=outputs/beyond4_lossless/clips/${CL}_origin_ll/${CL}_inpainting_results_sbs.mkv
  E=$(cut -d' ' -f1 "$M.md5")
  echo "expected_md5=$E" >> $O
  $PY scripts/distill/runs/skeptic1/verify_ll_v1.py $CL $M $E 2>&1 | grep -v Warning >> $O
done
echo DONE >> $O
