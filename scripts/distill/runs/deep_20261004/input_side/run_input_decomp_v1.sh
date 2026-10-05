#!/bin/bash
# probe 2 step A (CPU): input decomposition of every splat variant with R1's registration, 4 clips.
set -u
cd /home/kawa/master_project/StereoCrafter
PY=/home/kawa/miniconda3/envs/stereocrafter/bin/python
D=scripts/distill/runs/deep_20261004/input_side
OUT=$D/input_decomp_v1
mkdir -p $OUT
for c in 0301 0204 0052 0147; do
  RJ=outputs/deep_20261004/input_side/scores_inputonly_v1/${c}__R1.json
  until [ -f $RJ ]; do sleep 15; done
  for v in deployed R1 NA B1 B4 ZB SS4 SS4C; do until [ -f /mnt/ssd_data/deep_20261004/input_side/inputs_v1/$c/$v/params.json ]; do sleep 15; done; done
  CUDA_VISIBLE_DEVICES='' $PY $D/input_decomp_v1.py $RJ $OUT/${c}.json $c deployed R1 NA B1 B4 ZB SS4 SS4C
done
echo INPUT_DECOMP_DONE
