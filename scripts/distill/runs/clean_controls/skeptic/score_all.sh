#!/usr/bin/env bash
# Skeptic re-score of every lossless render on disk for one clip, via beyond4/score_clip_ll.py (SCORE_STEP=4).
# usage: bash score_all.sh <clip> <gpu>
set -euo pipefail
CLIP=$1; GPU=$2
cd /home/kawa/master_project/StereoCrafter
set +u; source ~/miniconda3/etc/profile.d/conda.sh; conda activate stereocrafter; set -u
export CUDA_VISIBLE_DEVICES=$GPU
export SCORE_STEP=4
SK=scripts/distill/runs/clean_controls/skeptic
specs=()
for p in \
  outputs/beyond4_lossless/clips/${CLIP}_origin_ll/${CLIP}_inpainting_results_sbs.mkv \
  outputs/beyond4_lossless/clips/${CLIP}_s25_ll/${CLIP}_inpainting_results_sbs.mkv \
  outputs/skeptic1_stack/clips/${CLIP}_student_ll/${CLIP}_inpainting_results_sbs.mkv \
  outputs/beyond_distil_mamba/clips/${CLIP}_mstudent1_step600_ll/${CLIP}_inpainting_results_sbs.mkv \
  scripts/distill/runs/clean_controls/selfA/infer/${CLIP}_*/${CLIP}_inpainting_results_sbs.mkv \
  scripts/distill/runs/clean_controls/regB/infer/*_${CLIP}/${CLIP}_inpainting_results_sbs.mkv \
  scripts/distill/runs/clean_controls/regB/infer/*_${CLIP}_*/${CLIP}_inpainting_results_sbs.mkv ; do
  if [ -f "$p" ]; then specs+=("${CLIP}=$p"); else echo "MISSING $p" >&2; fi
done
printf '%s\n' "${specs[@]}" > $SK/speclist_${CLIP}.txt
echo "scoring ${#specs[@]} renders for $CLIP on GPU $GPU"
python scripts/distill/runs/fulldata_v2/beyond4/score_clip_ll.py "${specs[@]}" 2> $SK/score_${CLIP}.stderr | tee $SK/scores_skeptic_${CLIP}.txt
echo "EXIT $?" >> $SK/scores_skeptic_${CLIP}.txt
