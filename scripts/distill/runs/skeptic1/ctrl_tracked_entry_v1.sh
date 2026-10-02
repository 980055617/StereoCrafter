#!/bin/bash
# Can the distilled deliverable ship through the TRACKED entry point (no hook script)?
# inpainting_inference.py --unet_state_path=step800.pt --expected_partial_unet_state=True
# must reproduce my hook run bit-for-bit.
set -u; cd /home/kawa/master_project/StereoCrafter
PY=/home/kawa/miniconda3/envs/stereocrafter/bin/python
until ! systemctl --user is-active --quiet sk-ext0 && ! systemctl --user is-active --quiet sk-ext1 \
      && ! systemctl --user is-active --quiet sk-gpu0 && ! systemctl --user is-active --quiet sk-gpu1 \
      && ! systemctl --user is-active --quiet sk-ctrl && ! systemctl --user is-active --quiet sk-oraclem; do sleep 20; done
OD=outputs/skeptic1_stack/control/0301_student_trackedentry; mkdir -p $OD
O=scripts/distill/runs/skeptic1/TRACKED_ENTRY.txt
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LOSSLESS_SBS=1 KEEP_ANAGLYPH=0 MAMBA_SELF_ATTN_INCLUDE='__nomatch__'
$PY scripts/distill/runs/fulldata_v2/beyond4/infer_lossless.py \
   --config=config/0160_overfit_inference_matched.json \
   --unet_state_path=scripts/distill/runs/beyond_distil/smoke1/step800.pt \
   --expected_partial_unet_state=True \
   --num_inference_steps=8 --min_guidance_scale=1.01 --max_guidance_scale=1.01 \
   --input_video_path=video_data/splatting/0301_splatting_results.mp4 --save_dir=$OD > $OD.log 2>&1
{
echo "=== does the tracked entry point load the 15-tensor deliverable and give the same pixels as the hook? ==="
date
grep -E "Partial UNet state load|missing=|unexpected=" $OD.log | head -3
echo "-- pre-encode array md5: tracked entry vs my hook run --"
grep _sbs $OD/writer_md5.txt
cat outputs/skeptic1_stack/clips/0301_student_ll/0301_inpainting_results_sbs.mkv.md5
} > $O 2>&1
echo TRACKED_DONE
