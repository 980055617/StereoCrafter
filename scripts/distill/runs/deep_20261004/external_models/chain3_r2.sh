#!/bin/bash
# D5: ONE GPU-0 lock hold: render the remaining 8 test clips (primary), then the DEV-only mask-threshold check
# (PREREG_ADDENDUM_r2_dev_mask.txt): primary and t95 on dev clips 0040 0082 0091.
set -u
cd /home/kawa/master_project/StereoCrafter
R=scripts/distill/runs/deep_20261004/external_models
PY=/mnt/ssd_data/deep_20261004/external_models/venv/bin/python
LOG=$R/chain_m2svid_fa_w16_r2.log
echo "CHAIN3_START $(date '+%F_%T')" >> $LOG
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True flock /tmp/claude-gpu0.lock bash -c "
  echo \"LOCK_ACQUIRED \$(date '+%F_%T')\" >> $LOG
  $PY $R/m2svid_infer_ll.py --tag m2svid_fa_w16 --win 16 --decode_chunk 8 --clips 0042 0125 0128 0141 0170 0225 0251 0259 >> $R/render_m2svid_fa_w16_B_r2.log 2>&1
  echo \"RENDER_B rc=\$? \$(date '+%F_%T')\" >> $LOG
  $PY $R/m2svid_infer_ll.py --tag m2svid_fa_w16 --win 16 --decode_chunk 8 --clips 0040 0082 0091 >> $R/render_dev_primary_r2.log 2>&1
  echo \"RENDER_DEV_PRIMARY rc=\$? \$(date '+%F_%T')\" >> $LOG
  $PY $R/m2svid_infer_ll.py --tag m2svid_fa_w16_t95 --win 16 --decode_chunk 8 --mask_thresh 0.95 --clips 0040 0082 0091 >> $R/render_dev_t95_r2.log 2>&1
  echo \"RENDER_DEV_T95 rc=\$? \$(date '+%F_%T')\" >> $LOG
"
echo "CHAIN3_END $(date '+%F_%T')" >> $LOG
