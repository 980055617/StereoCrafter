#!/bin/bash
# fulldata_fit.sh <point: 13|40|120|all|c13w14> <label> [extra env...]   -> runs/fulldata/fits/<label>.{json,best.pt,last.pt}
set -u; cd /home/kawa/master_project/StereoCrafter; D=scripts/distill; PY=/home/kawa/miniconda3/envs/stereocrafter/bin/python
POINT=$1; LABEL=$2; shift 2; F=$D/runs/fulldata/fits; mkdir -p $F
CACHE=/mnt/ssd_data/attn_cache/fulldata_tf
if [ "$POINT" = "c13w14" ]; then TRAIN=13; CACHE=/mnt/ssd_data/attn_cache/fulldata_tf:/mnt/ssd_data/attn_cache/fulldata_tf_c13w14; else TRAIN=$POINT; fi
echo "FIT_START $LABEL point=$POINT $(date +%H:%M:%S)"
env CACHE=$CACHE TRAIN=$TRAIN OUT=$F/$LABEL.json SAVE=$F/$LABEL "$@" $PY $D/distill_fulldata.py > $F/$LABEL.log 2>&1
echo "FIT_END $LABEL rc=$? $(date +%H:%M:%S)"; grep -E 'DONE|SUMMARY|Traceback|Error' $F/$LABEL.log | cut -c1-300
