#!/bin/bash
cd /home/kawa/master_project/StereoCrafter; D=scripts/distill; R=$D/runs/fulldata/temporal; export CUDA_VISIBLE_DEVICES=1
TEST=$(python3 -c "import json;print(' '.join(json.load(open('scripts/distill/splits/fulldata_v1.json'))['test']))")
ARGS=""; for C in $TEST 0160; do ARGS="$ARGS ${C}=outputs/fulldata/clips/${C}_origin/${C}_inpainting_results_sbs.mp4 ${C}=outputs/fulldata/clips/${C}_all_8k/${C}_inpainting_results_sbs.mp4"; done
echo "TEMPORAL_576 START $(date +%H:%M:%S)"; conda run -n stereocrafter --no-capture-output python3 $D/score_temporal.py $R/temporal_576.json $ARGS 2>&1 | grep -vE 'Warning|warn|Setting up|Loading model|/home/kawa' | tee $R/temporal_576.txt
HARGS=""; for C in 0160 0170 0204 0042 0052; do HARGS="$HARGS ${C}=outputs/fulldata/fullhd/${C}_origin/${C}_inpainting_results_sbs.mp4 ${C}=outputs/fulldata/fullhd/${C}_all_8k/${C}_inpainting_results_sbs.mp4"; done
echo "TEMPORAL_FULLHD START $(date +%H:%M:%S)"; conda run -n stereocrafter --no-capture-output python3 $D/score_temporal.py $R/temporal_fullhd.json $HARGS 2>&1 | grep -vE 'Warning|warn|Setting up|Loading model|/home/kawa' | tee $R/temporal_fullhd.txt
echo "TEMPORAL_DONE $(date +%H:%M:%S)"
