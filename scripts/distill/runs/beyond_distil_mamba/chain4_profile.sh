#!/bin/bash
# What option (B) gives up: attn1 UNet-MODULE time for origin / Mamba 2-slot (down0 only) / Mamba 5-slot,
# at the deployed 576x1024 crop and at the 1920x1024 stage where the published Mamba win is largest
# (-21.7 %).  Same instrument that produced the published -5.3 / -20.5 / -21.7 % numbers.
set -u
cd /home/kawa/master_project/StereoCrafter
PY=/home/kawa/miniconda3/envs/stereocrafter/bin/python
D=scripts/distill/runs/beyond_distil_mamba
OUT=outputs/beyond_distil_mamba/profile
until ! systemctl --user is-active --quiet bdm-chain3; do sleep 30; done
export CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LOSSLESS_SBS=0 KEEP_ANAGLYPH=0 PM_CHUNKS=3 PM_CLIP=0301
for RES in 576x1024 1024x1792 1920x1024; do
  H=${RES%x*}; W=${RES#*x}
  for CFG in none down0 all5; do
    OD=$OUT/${RES}_${CFG}
    [ -f "$OD/module_profile_${RES}_${CFG}.json" ] && { echo "SKIP $RES $CFG"; continue; }
    mkdir -p $OD
    PM_LABEL=${RES}_${CFG} PM_SLOTS=$CFG PM_H=$H PM_W=$W PM_OUT=$OD \
      $PY $D/profile_modules_v1.py > $OD/run.log 2>&1
    echo "PROFILE $RES $CFG rc=$? $(date +%T) dir=$OD"
  done
done
$PY $D/parse_profile_v1.py $D/PROFILE_ATTN1.txt $OUT
cat $D/PROFILE_ATTN1.txt
echo CHAIN4_DONE
