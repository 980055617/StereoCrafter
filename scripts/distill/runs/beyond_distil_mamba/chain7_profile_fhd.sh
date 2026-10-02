#!/bin/bash
# The Full-HD stage, with the axes the right way round.  The project labels it "1920x1024" meaning
# WIDTH x HEIGHT, and clip 0301's source is h=1024 w=1920, so inpainting_inference wants
# target_height=1024 target_width=1920.  chain4's 1920x1024 attempt passed them swapped and was
# correctly refused by _center_crop_frames ("Requested crop 1920x1024 exceeds source 1024x1920").
set -u
cd /home/kawa/master_project/StereoCrafter
PY=/home/kawa/miniconda3/envs/stereocrafter/bin/python
D=scripts/distill/runs/beyond_distil_mamba
OUT=outputs/beyond_distil_mamba/profile_fhd
until ! systemctl --user is-active --quiet bdm-chain6; do sleep 20; done
export CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LOSSLESS_SBS=0 KEEP_ANAGLYPH=0 PM_CHUNKS=3 PM_CLIP=0301
for CFG in none down0 all5; do
  OD=$OUT/1024x1920_${CFG}
  k=1; while [ -d "$OD" ]; do k=$((k+1)); OD=$OUT/1024x1920_${CFG}_r$k; done
  mkdir -p $OD
  PM_LABEL=1024x1920_${CFG} PM_SLOTS=$CFG PM_H=1024 PM_W=1920 PM_OUT=$OD \
    $PY $D/profile_modules_v1.py > $OD/run.log 2>&1
  echo "PROFILE_FHD $CFG rc=$? $(date +%T) dir=$OD"
done
$PY $D/parse_profile_v2.py $D/PROFILE_ATTN1_fhd.txt $OUT
cat $D/PROFILE_ATTN1_fhd.txt
echo CHAIN7_DONE $(date +%T)
