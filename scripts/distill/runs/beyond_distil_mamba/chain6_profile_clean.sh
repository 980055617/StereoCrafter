#!/bin/bash
# RE-MEASURE the 576x1024 attn1 timings with the GPU to ourselves.  The first pass (profile/) had its
# origin baseline overlap the tail of an LPIPS scoring job, and a contended baseline would overstate the
# Mamba saving.  New directory, the first pass is kept for comparison.  Also 5 chunks instead of 3, so
# the one-time Triton JIT warmup is amortised over 40 calls per module instead of 24.
set -u
cd /home/kawa/master_project/StereoCrafter
PY=/home/kawa/miniconda3/envs/stereocrafter/bin/python
D=scripts/distill/runs/beyond_distil_mamba
OUT=outputs/beyond_distil_mamba/profile2
until ! systemctl --user is-active --quiet bdm-chain4; do sleep 20; done
export CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LOSSLESS_SBS=0 KEEP_ANAGLYPH=0 PM_CHUNKS=5 PM_CLIP=0301
for RES in 576x1024; do
  H=${RES%x*}; W=${RES#*x}
  for CFG in none down0 all5 none down0 all5; do
    OD=$OUT/${RES}_${CFG}
    k=1; while [ -d "$OD" ]; do k=$((k+1)); OD=$OUT/${RES}_${CFG}_r$k; done
    mkdir -p $OD
    PM_LABEL=${RES}_${CFG} PM_SLOTS=$CFG PM_H=$H PM_W=$W PM_OUT=$OD \
      $PY $D/profile_modules_v1.py > $OD/run.log 2>&1
    echo "PROFILE2 $RES $CFG rc=$? $(date +%T) dir=$OD"
  done
done
$PY $D/parse_profile_v2.py $D/PROFILE_ATTN1_clean.txt $OUT
cat $D/PROFILE_ATTN1_clean.txt
echo CHAIN6_DONE $(date +%T)
