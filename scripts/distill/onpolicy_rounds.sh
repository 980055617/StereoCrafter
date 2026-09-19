#!/bin/bash
# DAgger-style on-policy distillation rounds for the 5 light level-0 slots:
#   run the full STUDENT model on 0160, capture (student-cascade input x, origin_attn(x), time_emb),
#   train all slots on those pairs (init = previous round), inject onto origin, read LPIPS. Repeat.
cd /home/kawa/master_project/StereoCrafter
S=/home/kawa/master_project/StereoCrafter/scripts/distill
PY=/home/kawa/miniconda3/envs/stereocrafter/bin/python; R=$S/runs
until grep -q 'LONG_DISTILL_DONE' $S/long_distill.log 2>/dev/null; do sleep 60; done
INIT=$R/long6k.pt; [ -f $INIT ] || INIT=$R/e210_asis.pt
for ROUND in 1 2 3; do
  CACHE=/mnt/ssd_data/attn_cache/0160_onpolicy_r$ROUND; rm -rf $CACHE
  echo "ROUND $ROUND CAPTURE_START $(date +%H:%M:%S) init=$INIT"
  CAP_ONPOLICY=1 CAP_OUT=$CACHE CAP_KEEP=4 CKPT=$INIT $PY $S/capture_attn.py > $R/onpolicy_r${ROUND}_capture.log 2>&1
  grep -E 'ON-POLICY|DONE|Traceback|Error' $R/onpolicy_r${ROUND}_capture.log | tail -3
  echo "ROUND $ROUND TRAIN_START $(date +%H:%M:%S)"
  CACHE=$CACHE CKPT=$INIT STEPS=2000 LR=3e-4 BATCH=8 EVAL_EVERY=500 OUT=$R/onpolicy_r$ROUND.json SAVE=$R/onpolicy_r$ROUND.pt $PY $S/distill_standalone.py > $R/onpolicy_r$ROUND.log 2>&1
  grep -E 'SUMMARY|Traceback' $R/onpolicy_r$ROUND.log
  echo "ROUND $ROUND EVAL_START $(date +%H:%M:%S)"
  $S/eval_light.sh $R/onpolicy_r$ROUND.pt origin_plus_onpolicy_r$ROUND 2>&1 | grep -E "LPIPS|origin_plus_onpolicy_r$ROUND |^RESULT"
  INIT=$R/onpolicy_r$ROUND.pt
done
echo "ONPOLICY_DONE $(date +%H:%M:%S)"
