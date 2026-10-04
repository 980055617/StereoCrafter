#!/bin/bash
# judge J1c (LPIPS, score_clip_ll.py UNCHANGED, SCORE_STEP=4) and J1d (warp, validate lane's score_temporal_ll.py
# UNCHANGED, STEP unset) for the temporal lane's dcs14 key rows, one scorer call per clip, GPU 1 under its lock
# (held per scorer call).  Outputs only under outputs/more_20261004/judge/ and this directory.  Never overwrites.
set -u
cd /home/kawa/master_project/StereoCrafter
PY=/home/kawa/miniconda3/envs/stereocrafter/bin/python
J=scripts/distill/runs/more_20261004/judge
O=outputs/more_20261004/judge/rescore_temporal
mkdir -p $O
export CUDA_VISIBLE_DEVICES=1
for C in 0170 0259 0042 0301; do
  SPECS="$C=outputs/beyond4_lossless/clips/${C}_origin_ll/${C}_inpainting_results_sbs.mkv $C=outputs/finalcheck_20261004/speed/clips/${C}_deliv_g100_T5nat/${C}_inpainting_results_sbs.mkv $C=outputs/more_20261004/temporal/clips/${C}_T5nat_dcs14/${C}_inpainting_results_sbs.mkv $C=outputs/more_20261004/temporal/clips/${C}_origin_g101_s8_dcs14/${C}_inpainting_results_sbs.mkv"
  for s in $SPECS; do [ -f "${s#*=}" ] || { echo "MISSING ${s#*=}"; exit 4; }; done
  OUTF=$J/SCORES_J1c_LPIPS_$C.txt
  if [ -e "$OUTF" ]; then echo "refusing to overwrite $OUTF"; else
    echo "SCORE_START $(date +%F_%T) scorer=scripts/distill/runs/fulldata_v2/beyond4/score_clip_ll.py SCORE_STEP=4" > $OUTF
    SCORE_STEP=4 flock /tmp/claude-gpu1.lock $PY scripts/distill/runs/fulldata_v2/beyond4/score_clip_ll.py $SPECS >> $OUTF 2>&1
    echo "SCORE_DONE rc=$? $(date +%F_%T)" >> $OUTF
  fi
  OUTJ=$O/J1d_warp_$C.json; LOG=$O/J1d_warp_$C.log
  if [ -e "$OUTJ" ] || [ -e "$LOG" ]; then echo "refusing to overwrite $OUTJ"; else
    echo "TSCORE_START $(date +%F_%T) scorer=scripts/distill/runs/finalcheck_20261004/validate/score_temporal_ll.py" > $LOG
    (unset STEP; flock /tmp/claude-gpu1.lock $PY scripts/distill/runs/finalcheck_20261004/validate/score_temporal_ll.py $OUTJ $SPECS) >> $LOG 2>&1
    echo "TSCORE_DONE rc=$? $(date +%F_%T)" >> $LOG
  fi
done
echo "J1_SCORING_DONE $(date +%F_%T)"
