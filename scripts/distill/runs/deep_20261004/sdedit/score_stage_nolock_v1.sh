#!/bin/bash
# deep_20261004 / sdedit lane: NO-LOCK variant of score_stage_v1.sh for batch mode -- identical except that it does NOT
# take /tmp/claude-gpu0.lock itself: it must only be run by a batch script that already holds that lock.  Every stage writes a NEW directory and refuses to reuse one.
# usage: score_stage_v1.sh <tag> <clip> [<clip> ...]
#   rows per clip: deployed origin_ll / mstudent2_step800_deliv_ll, guidance controls origin_g100_s8 / deliv_g100_s8,
#   every existing lane render outputs/deep_20261004/sdedit/clips/<clip>_{origin,deliv}_g100_{sd31,sd7,sd1,sd1fill},
#   and the descriptive INPUT rows if they exist (outputs/deep_20261004/sdedit/input_rows/clips/<clip>_INPUT_{warp,fill}).
#   M1  score_clip_ll.py (unchanged), per clip         -> SCORES.txt (appended)
#   M1r score_registered_v1.py (unchanged) per clip     -> reg/<clip>.json   (0125: score_registered_v1w.py, wide grid)
#   M2  run_decomp_sdedit_v1.py --input (CPU)           -> decomp/decomp.json, DECOMP.txt
#   M3  score_temporal_ll.py (unchanged), per clip      -> temporal/<clip>.json, merged temporal.json
#   K3  check_passthrough_v1.py (unchanged, CPU)        -> K3_passthrough.txt
set -u
cd /home/kawa/master_project/StereoCrafter
L=scripts/distill/runs/deep_20261004/sdedit
O=outputs/deep_20261004/sdedit
PY=/home/kawa/miniconda3/envs/stereocrafter/bin/python
LOCK=/tmp/claude-gpu0.lock
TAG=$1; shift
D=$O/score_$TAG
if [ -e $D ]; then echo "refusing to reuse $D"; exit 3; fi
mkdir -p $D/reg $D/decomp $D/temporal
echo "STAGE $TAG clips $* start $(date +%F_%T)" | tee -a $D/stage.log
: > $D/scorelist.txt; : > $D/specs.txt; : > $D/renderdirs.txt
for c in "$@"; do
  for spec in "origin_ll=outputs/beyond4_lossless/clips/${c}_origin_ll" \
              "mstudent2_step800_deliv_ll=outputs/beyond_distil_mamba_scaled/clips/${c}_mstudent2_step800_deliv_ll" \
              "origin_g100_s8=outputs/finalcheck_20261004/speed/clips/${c}_origin_g100_s8" \
              "deliv_g100_s8=outputs/finalcheck_20261004/speed/clips/${c}_deliv_g100_s8"; do
    lab=${spec%%=*}; dir=${spec#*=}
    f=$dir/${c}_inpainting_results_sbs.mkv
    [ -f $f ] || { echo "MISSING baseline $f" | tee -a $D/stage.log; exit 4; }
    echo "$c=$f" >> $D/scorelist.txt; echo "$c:$lab=$f" >> $D/specs.txt
  done
  for m in origin deliv; do for v in sd31 sd7 sd1 sd1fill; do
    dir=$O/clips/${c}_${m}_g100_${v}; f=$dir/${c}_inpainting_results_sbs.mkv
    if [ -f $f ]; then echo "$c=$f" >> $D/scorelist.txt; echo "$c:${m}_g100_${v}=$f" >> $D/specs.txt; echo $dir >> $D/renderdirs.txt; fi
  done; done
  for k in warp fill; do
    dir=$O/input_rows/clips/${c}_INPUT_${k}; f=$dir/${c}_inpainting_results_sbs.mkv
    if [ -f $f ]; then echo "$c=$f" >> $D/scorelist.txt; echo "$c:INPUT_${k}=$f" >> $D/specs.txt; echo $dir >> $D/renderdirs.txt; fi
  done
done
echo "rows: $(wc -l < $D/scorelist.txt)" | tee -a $D/stage.log
# ---- K3 pass-through (CPU)
CUDA_VISIBLE_DEVICES= $PY scripts/distill/runs/finalcheck_20261004/speed/check_passthrough_v1.py $D/K3_passthrough.txt \
  $D/K3_ref_cache.json $(cat $D/renderdirs.txt | tr '\n' ' ') > $D/K3_passthrough.log 2>&1
echo "K3 rc=$? $(tail -1 $D/K3_passthrough.txt 2>/dev/null)" | tee -a $D/stage.log
# ---- M2 decomposition (CPU only) runs in the background while the GPU steps queue for the lock
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 nice -n 10 $PY $L/run_decomp_sdedit_v1.py $D/decomp/decomp.json $D/decomp/DECOMP.txt \
  --input $(cat $D/specs.txt | tr '\n' ' ') > $D/decomp/run.log 2>&1 &
M2PID=$!
# ---- M1, M1r, M3: all GPU steps under ONE acquisition of the GPU-0 lock (score_gpu_steps_v1.sh, no flock inside)
bash $L/score_gpu_steps_v1.sh $D "$@"      # the CALLER holds /tmp/claude-gpu0.lock (batch mode)
echo "GPU steps rc=$?" | tee -a $D/stage.log
# ---- M2 decomposition (CPU, started in the background right after K3; collected here)
wait $M2PID
echo "M2 rc=$?" | tee -a $D/stage.log
echo "STAGE $TAG done $(date +%F_%T)" | tee -a $D/stage.log
