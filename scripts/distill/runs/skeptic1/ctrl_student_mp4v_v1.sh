#!/bin/bash
# FAITHFULNESS of MY OWN student runner: same wrapper with LOSSLESS_SBS=0 must reproduce the
# step-distillation lane's mp4v student output for 0301 BYTE FOR BYTE.
set -u; cd /home/kawa/master_project/StereoCrafter
PY=/home/kawa/miniconda3/envs/stereocrafter/bin/python
until ! systemctl --user is-active --quiet sk-ext0 && ! systemctl --user is-active --quiet sk-ext1 \
      && ! systemctl --user is-active --quiet sk-gpu0 && ! systemctl --user is-active --quiet sk-gpu1; do sleep 20; done
OD=outputs/skeptic1_stack/control/0301_student_mp4v; mkdir -p $OD
O=scripts/distill/runs/skeptic1/FAITHFULNESS_skeptic.txt
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LOSSLESS_SBS=0 KEEP_ANAGLYPH=0 MAMBA_SELF_ATTN_INCLUDE='__nomatch__'
export SK_CLIP=0301 SK_OUT=$OD SK_STEPS=8 SK_GUID=1.01 SK_UNET='' SK_CK=scripts/distill/runs/beyond_distil/smoke1/step800.pt
$PY scripts/distill/runs/skeptic1/infer_ll_hook.py > $OD.log 2>&1
{
echo "=== skeptic lane faithfulness: my student runner vs the step-distillation lane's ==="
date
echo "-- md5 of the mp4v files (mine, with LOSSLESS_SBS=0, vs theirs from chain2) --"
md5sum $OD/0301_inpainting_results_sbs.mp4 outputs/beyond_distil/0301_smoke1_step800/0301_inpainting_results_sbs.mp4
echo "-- my pre-encode array md5, mp4v control vs my lossless run (must be identical) --"
grep _sbs $OD/writer_md5.txt
cat outputs/skeptic1_stack/clips/0301_student_ll/0301_inpainting_results_sbs.mkv.md5
echo "-- swap log --"
grep -E '^\[sk\]' $OD.log
} > $O 2>&1
echo CTRL_DONE
