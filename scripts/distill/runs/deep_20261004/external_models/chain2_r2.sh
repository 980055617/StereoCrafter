#!/bin/bash
# Replacement for chain_r2.sh after RENDER_A (D2/D3): ONE GPU-0 lock hold renders the remaining 8 test clips.
# Scoring runs on CPU (score_cpu_r2.sh, no lock), gated on reproducing the published origin / deliverable numbers.
set -u
cd /home/kawa/master_project/StereoCrafter
R=scripts/distill/runs/deep_20261004/external_models
PY=/mnt/ssd_data/deep_20261004/external_models/venv/bin/python
TAG=m2svid_fa_w16
LOG=$R/chain_${TAG}_r2.log
B="0042 0125 0128 0141 0170 0225 0251 0259"
echo "CHAIN2_START $(date '+%F_%T')" >> $LOG
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True flock /tmp/claude-gpu0.lock \
  $PY $R/m2svid_infer_ll.py --tag $TAG --win 16 --decode_chunk 8 --clips $B >> $R/render_${TAG}_B_r2.log 2>&1
echo "RENDER_B rc=$? $(date '+%F_%T')" >> $LOG
echo "CHAIN2_END $(date '+%F_%T')" >> $LOG
