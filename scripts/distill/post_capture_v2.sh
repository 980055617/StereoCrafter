#!/bin/bash
# after the v2 capture: fit all_8k on the corrected cache (fresh init, standard recipe), evaluate vs real GT, on-policy check
set -u; cd /home/kawa/master_project/StereoCrafter; D=scripts/distill; R=$D/runs/fulldata_v2; F=$R/fits
set +u; source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate stereocrafter; set -u
until grep -q CAPTURE_V2_DONE $R/capture_lane.log 2>/dev/null; do sleep 300; done
grep -qE 'CHUNK_FAIL|DISK_LOW' $R/capture_lane.log && { echo "CAPTURE_PROBLEM -- not fitting"; exit 1; }
echo "FIT_START $(date +%H:%M:%S)"
CUDA_VISIBLE_DEVICES=0 CACHE=/mnt/ssd_data/attn_cache/fulldata_tf_v2 TRAIN=all STEPS=8000 LR=5e-4 WARMUP=200 BATCH=8 EVAL_EVERY=1000 NW=0 OUT=$F/all_8k_v2.json SAVE=$F/all_8k_v2 python $D/distill_fulldata.py > $F/all_8k_v2.log 2>&1
echo "FIT_END rc=$? $(date +%H:%M:%S)"; grep -E 'DONE|SUMMARY|Traceback' $F/all_8k_v2.log | cut -c1-300
cp $F/all_8k_v2.best.pt /mnt/ssd_data/stereocrafter_weights/_distill_injected/light_lvl0_fulldata333_v2_8k_mamba_only.pt
$D/eval_v2.sh $F/all_8k_v2.best.pt all_8k_v2 2>&1 | grep -E 'REALGT_SUMMARY|WARN|Traceback|EVAL_V2_DONE'
echo "POST_CAPTURE_V2_DONE $(date +%H:%M:%S)"
