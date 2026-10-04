#!/bin/bash
# ays_20261004/ays5 step-3 batch (PREREG.txt STEP 3 + ADDENDUM 2).  Run ONLY as
#     flock /tmp/claude-gpu0.lock bash scripts/distill/runs/ays_20261004/ays5/batch_S3_v2.sh
# The caller holds the GPU-0 lock for the whole batch; the v2 driver / scorer have no inner flock.
# remaining 3a + all 3b renders -> S3 pairing/config check (stop on fail) -> S3 scoring (12 clips) -> analysis v2.
set -u
set +u; source ~/miniconda3/etc/profile.d/conda.sh; conda activate stereocrafter; set -u
cd /home/kawa/master_project/StereoCrafter
L=scripts/distill/runs/ays_20261004/ays5
CK=/mnt/ssd_data/stereocrafter_weights/_distill_injected/mamba5slot_plus_stepdistil_up3_train10clip_step800_20261001.pt
if flock -n /tmp/claude-gpu0.lock true; then echo "NOT UNDER /tmp/claude-gpu0.lock -> abort (run via flock)"; exit 7; fi
echo "BATCH_S3_START $(date +%F_%T) ckpt_md5=$(md5sum $CK | cut -d' ' -f1)"
bash $L/run_driver_ays5_nolock_v2.sh $L/jobs_S3_batch_v2.txt 0
python $L/check_pairing_ays5_v2.py S3 $L/PAIRING_S3.txt || { echo "PAIRING_S3_FAIL -> STOP before scoring $(date +%F_%T)"; echo "BATCH_S3_STOPPED"; exit 6; }
echo "PAIRING_S3_PASS $(date +%F_%T)"
bash $L/score_ays5_nolock_v2.sh S3 S3 0042 0052 0125 0128 0141 0147 0170 0204 0225 0251 0259 0301
python $L/analyze_ays5_v2.py $L/TABLE_S3_AYS5_FULL.txt $L/TABLE_S3_AYS5_FULL.json
echo "BATCH_S3_DONE $(date +%F_%T) ckpt_md5=$(md5sum $CK | cut -d' ' -f1)"
