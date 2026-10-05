#!/bin/bash
# origin on the 5 remaining dev clips + control C2 (25-step hook path vs existing s25 renders), then UNREG scores of the 6 dev
# origin renders.  Each GPU job takes the GPU-1 lock separately.
set -u
cd /home/kawa/master_project/StereoCrafter
set +u; source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate stereocrafter; set -u
L=scripts/distill/runs/deep_20261004/scale_gt
export CUDA_VISIBLE_DEVICES=1
echo "ORIGIN_C2_START $(date +%F_%T)"
bash $L/run_render_v1.sh $L/jobs_origin_dev_c2_v1.txt
ARGS=""
for c in 0040 0082 0091 0184 0245 0268; do ARGS="$ARGS $c=outputs/deep_20261004/scale_gt/renders/${c}_origin_s8/${c}_inpainting_results_sbs.mkv"; done
SCORE_STEP=4 flock /tmp/claude-gpu1.lock python scripts/distill/runs/fulldata_v2/beyond4/score_clip_ll.py $ARGS > $L/score_dev_origin_s8.txt 2>&1 < /dev/null
echo "SCORED origin dev: $(grep -c '^ROW' $L/score_dev_origin_s8.txt) ROWs $(date +%F_%T)"
for c in 0301 0042; do
  A=$(cut -d' ' -f1 outputs/deep_20261004/scale_gt/renders/${c}_origin_s25_ctlC2/writer_md5.txt | head -1)
  if [ $c = 0301 ]; then B=$(cut -d' ' -f1 outputs/beyond4_lossless/clips/0301_s25_ll/writer_md5.txt | head -1); else B=$(cut -d' ' -f1 outputs/skeptic1_stack/clips/0042_s25_ll/writer_md5.txt | head -1); fi
  if [ "$A" = "$B" ]; then echo "C2 $c PASS md5 $A"; else echo "C2 $c FAIL hook=$A existing=$B"; fi
done
echo "ORIGIN_C2_DONE $(date +%F_%T)"
