#!/bin/bash
# after the per-slot noise test on GPU 1: attention-concentration stats on 3 clips (2 windows each) + at 1024x1792
cd /home/kawa/master_project/StereoCrafter; D=scripts/distill; PY=/home/kawa/miniconda3/envs/stereocrafter/bin/python
BEST=/mnt/ssd_data/stereocrafter_weights/_distill_injected/light_lvl0_multiclip13_r2_mamba_only.pt
until grep -q 'NOISE_SLOT_DONE' $D/runs/noise_slot.log 2>/dev/null; do sleep 30; done
export CUDA_VISIBLE_DEVICES=1
for C in 0160 0042 0021; do O=$D/runs/attn_stats/$C; mkdir -p $O
  echo "ATTN_STATS $C"; CAP_OUT=$O CKPT=$BEST CAP_MAXCHUNKS=3 CAP_VIDEO=video_data/splatting/${C}_splatting_results.mp4 $PY $D/attn_stats.py 2>&1 | grep -vE 'Warning|warn|it/s|Loaded|Materialized|\[info\]|\[sched\]|UNet config|filter' ; done
O=$D/runs/attn_stats/0160_1024x1792; mkdir -p $O; echo "ATTN_STATS 0160 @1024x1792"
CAP_OUT=$O CKPT=$BEST CAP_MAXCHUNKS=2 CAP_RES=1024x1792 $PY $D/attn_stats.py 2>&1 | grep -vE 'Warning|warn|it/s|Loaded|Materialized|\[info\]|\[sched\]|UNet config|filter'
echo "LANE2_DONE $(date +%H:%M:%S)"
