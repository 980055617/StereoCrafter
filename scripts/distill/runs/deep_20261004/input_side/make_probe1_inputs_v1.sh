#!/bin/bash
# input_side lane: build the probe-1 inputs of one clip from its fit JSON (PREREG: RF = fine-stage best (s*, o*),
# RO = s 1, o = the fine-stage optimum offset at s = 1).  CPU only.  usage: make_probe1_inputs_v1.sh <clip>
set -eu
cd /home/kawa/master_project/StereoCrafter
PY=/home/kawa/miniconda3/envs/stereocrafter/bin/python
FIT=/mnt/ssd_data/deep_20261004/input_side/fit_v1/fit_$1.json
RS=scripts/distill/runs/deep_20261004/input_side/resplat_v1.py
read S O O1 <<< "$($PY -c "import json;j=json.load(open('$FIT'));print(j['fit']['s'], j['fit']['o'], j['deployed_mapping_fine']['o'])")"
echo "clip $1: RF s=$S o=$O ; RO s=1 o=$O1   (from $FIT)"
CUDA_VISIBLE_DEVICES='' RS_THREADS=${RS_THREADS:-4} $PY $RS $1 RF s=$S o=$O
CUDA_VISIBLE_DEVICES='' RS_THREADS=${RS_THREADS:-4} $PY $RS $1 RO s=1 o=$O1
