#!/bin/bash
# Lane 1 (GPU 0): teacher-forced capture on the ORIGIN UNet (clip 0160) + architecture variant sweep on that cache.
# The 2026-09-13 sweep used a cache from the e210 UNet (wrong time_emb); this re-runs it on the deployed teacher.
set -u; cd /home/kawa/master_project/StereoCrafter
D=scripts/distill; PY=/home/kawa/miniconda3/envs/stereocrafter/bin/python; R=$D/runs/origin_sweep; mkdir -p $R
BEST=/mnt/ssd_data/stereocrafter_weights/_distill_injected/light_lvl0_multiclip13_r2_mamba_only.pt
CACHE=/mnt/ssd_data/attn_cache/0160_origin_tf
export CUDA_VISIBLE_DEVICES=0
if [ ! -f $CACHE/capture_meta.json ]; then
  echo "CAPTURE_START $(date +%H:%M:%S)"
  CAP_GATE=0.0 CAP_OUT=$CACHE CAP_KEEP=4 CKPT=$BEST $PY $D/capture_attn.py > $R/capture.log 2>&1
  echo "CAPTURE_END rc=$? files=$(ls $CACHE/*.pt | wc -l) $(du -sh $CACHE | cut -f1)"
fi
run() { local L=$1; shift; echo "PROBE_START $L $(date +%H:%M:%S)"
  env CACHE=$CACHE OUT=$R/$L.json "$@" $PY $D/distill_standalone.py > $R/$L.log 2>&1
  echo "PROBE_END $L rc=$?"; grep -E 'SUMMARY|Traceback' $R/$L.log | cut -c1-600; }
run best13_asis     CKPT=$BEST STEPS=1500 LR=3e-4 BATCH=8
run fresh_d128_fwd  CKPT=fresh STEPS=1500 LR=1e-3 BATCH=8
run linear_baseline CKPT=fresh KIND=linear STEPS=600 LR=1e-3 BATCH=8
run fresh_d64_fwd   CKPT=fresh D_STATE=64 STEPS=1500 LR=1e-3 BATCH=8
run fresh_d32_fwd   CKPT=fresh D_STATE=32 STEPS=1500 LR=1e-3 BATCH=8
run fresh_d256_fwd  CKPT=fresh D_STATE=256 STEPS=1500 LR=1e-3 BATCH=8
run fresh_d128_both CKPT=fresh BIDIR=both STEPS=1500 LR=1e-3 BATCH=8
run fresh_h32_fwd   CKPT=fresh HEADDIM=32 STEPS=1500 LR=1e-3 BATCH=8
run fresh_e2_fwd    CKPT=fresh EXPAND=2 STEPS=1500 LR=1e-3 BATCH=8
echo "SWEEP_ORIGIN_DONE $(date +%H:%M:%S)"
