#!/bin/bash
# scale_gt MAIN chain (after the smoke passed):
#   wait for phase A -> build specs (PREREG exclusions) -> encode latent cache (4 locked chunks) -> train MAIN (3000) ->
#   train CONTRAST (1000) -> Stage-1 dev renders (each render its own lock) + UNREG monitoring scores per checkpoint.
# Every GPU step runs under flock /tmp/claude-gpu1.lock with CUDA_VISIBLE_DEVICES=1.
set -u
cd /home/kawa/master_project/StereoCrafter
set +u; source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate stereocrafter; set -u
L=scripts/distill/runs/deep_20261004/scale_gt
C=/mnt/ssd_data/deep_20261004/scale_gt
export CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "MAIN_CHAIN_START $(date +%F_%T)"
until grep -q PREP_DONE $L/chain_prep_v3.log 2>/dev/null; do sleep 30; done
echo "PREP_DONE seen $(date +%F_%T); clips with clip.json: $(ls $C/cache_v1/crops/*/clip.json | wc -l)"
python $L/build_specs_v1.py $C/cache_v1/crops $C/cache_v1/latents $L > $L/build_specs_v1.log 2>&1
echo "SPECS rc=$? $(date +%F_%T)"
python - <<'EOF' > $L/encode_chunks_v1.txt
import json
o = json.load(open("scripts/distill/runs/deep_20261004/scale_gt/train_clips_v1.json"))["clips"]
n = len(o); k = 4
for i in range(k): print(",".join(o[i * n // k:(i + 1) * n // k]))
EOF
i=0
while read -r CL; do
  i=$((i+1))
  ENC_SPEC=$L/spec_main_v1.json,$L/spec_contrast8_v1.json flock /tmp/claude-gpu1.lock python $L/encode_cache_v1.py $C/cache_v1/crops $C/cache_v1/latents "$CL" > $L/encode_v1_chunk$i.log 2>&1 < /dev/null
  echo "ENCODE chunk $i rc=$? $(date +%F_%T)"
done < $L/encode_chunks_v1.txt
echo "LATENTS $(ls $C/cache_v1/latents/*.pt | wc -l) files"
flock /tmp/claude-gpu1.lock python $L/train_scale_gt_v1.py $L/spec_main_v1.json > $L/train_main_v1.log 2>&1
echo "TRAIN_MAIN rc=$? $(date +%F_%T)"
flock /tmp/claude-gpu1.lock python $L/train_scale_gt_v1.py $L/spec_contrast8_v1.json > $L/train_contrast8_v1.log 2>&1
echo "TRAIN_CONTRAST rc=$? $(date +%F_%T)"
DEV="0040 0082 0091 0184 0245 0268"
for RS in main_v1:500 main_v1:1000 contrast8_v1:500 contrast8_v1:1000 main_v1:2000 main_v1:3000 main_v1:1500 main_v1:250 contrast8_v1:250 main_v1:2500; do
  RUN=${RS%%:*}; S=${RS##*:}
  J=$L/jobs_dev_${RUN}_s${S}.txt
  : > $J
  for c in $DEV; do echo "$c ${RUN}_s${S}_s8 $C/ck/$RUN/step$S.pt 8" >> $J; done
  bash $L/run_render_v1.sh $J
  ARGS=""
  for c in $DEV; do ARGS="$ARGS $c=outputs/deep_20261004/scale_gt/renders/${c}_${RUN}_s${S}_s8/${c}_inpainting_results_sbs.mkv"; done
  SCORE_STEP=4 flock /tmp/claude-gpu1.lock python scripts/distill/runs/fulldata_v2/beyond4/score_clip_ll.py $ARGS > $L/score_dev_${RUN}_s${S}.txt 2>&1
  echo "DEV ${RUN} s${S} rendered+scored $(date +%F_%T) :: $(grep -c '^ROW' $L/score_dev_${RUN}_s${S}.txt) ROWs"
done
echo "MAIN_CHAIN_DONE $(date +%F_%T)"
