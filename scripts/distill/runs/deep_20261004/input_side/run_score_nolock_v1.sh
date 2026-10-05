#!/bin/bash
# input_side lane, NOLOCK variant (caller holds /tmp/claude-gpu0.lock): run score_input_v1.py for a list of (clip, input, label=path ...) lines, each under the GPU-0 lock.
# usage: run_score_v1.sh <listfile> <scores_dir>      list lines:  CLIP INPUT [label=path ...]
set -u
cd /home/kawa/master_project/StereoCrafter
if flock -n /tmp/claude-gpu0.lock true; then echo "NOT UNDER /tmp/claude-gpu0.lock -> abort"; exit 7; fi
PY=/home/kawa/miniconda3/envs/stereocrafter/bin/python
SC=scripts/distill/runs/deep_20261004/input_side/score_input_v1.py
INROOT=/mnt/ssd_data/deep_20261004/input_side/inputs_v1
LIST=$1; SD=$2
mkdir -p $SD
export CUDA_VISIBLE_DEVICES=0 SCORE_STEP=4
while read -r CLIP INPUT REST; do
  [ -z "${CLIP:-}" ] && continue
  case "$CLIP" in \#*) continue;; esac
  if [ -e "$SD/${CLIP}__${INPUT}.json" ]; then echo "SKIP $CLIP $INPUT exists"; continue; fi
  T0=$(date +%s)
  $PY $SC $SD $CLIP $INROOT/$CLIP/$INPUT $REST > $SD/${CLIP}__${INPUT}.log 2>&1
  echo "SCORE $CLIP $INPUT rc=$? secs=$(( $(date +%s) - T0 )) $(date +%H:%M:%S)"
done < "$LIST"
echo "SCORE_LIST_DONE $LIST $(date +%H:%M:%S)"
