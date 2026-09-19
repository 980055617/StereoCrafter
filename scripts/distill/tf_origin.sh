#!/bin/bash
# Clean teacher-forced distillation against the ORIGIN UNet (gate 0 => attention outputs, origin time_emb),
# fresh init and long6k init, then LPIPS on origin. Runs after the noise-sensitivity calibration.
cd /home/kawa/master_project/StereoCrafter
S=/home/kawa/master_project/StereoCrafter/scripts/distill
PY=/home/kawa/miniconda3/envs/stereocrafter/bin/python; R=$S/runs
until grep -q 'NOISE_SENS_DONE' $S/noise_sens.log 2>/dev/null; do sleep 60; done
CACHE=/mnt/ssd_data/attn_cache/0160_origin_tf; rm -rf $CACHE
echo "TF_ORIGIN CAPTURE_START $(date +%H:%M:%S)"
CAP_GATE=0.0 CAP_OUT=$CACHE CAP_KEEP=4 CKPT=$R/long6k.pt $PY $S/capture_attn.py > $R/tf_origin_capture.log 2>&1
grep -E 'DONE|Traceback|Error' $R/tf_origin_capture.log | tail -2
for V in fresh long6k; do
  CK=fresh; [ $V = long6k ] && CK=$R/long6k.pt
  echo "TF_ORIGIN TRAIN_START $V $(date +%H:%M:%S)"
  CACHE=$CACHE CKPT=$CK STEPS=3000 LR=5e-4 BATCH=8 EVAL_EVERY=500 OUT=$R/tf_origin_$V.json SAVE=$R/tf_origin_$V.pt $PY $S/distill_standalone.py > $R/tf_origin_$V.log 2>&1
  grep -E 'SUMMARY|Traceback' $R/tf_origin_$V.log
  $S/eval_light.sh $R/tf_origin_$V.pt origin_plus_tf_$V 2>&1 | grep -E "LPIPS|origin_plus_tf_$V |^RESULT"
done
echo "TF_ORIGIN_DONE $(date +%H:%M:%S)"
