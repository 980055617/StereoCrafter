#!/bin/bash
# ays_20261004/ays5 scoring.  Derived from scripts/distill/runs/more_20261004/judge/score_J3c_ays_clips_v1.sh (read-only
# source); the scorer itself is scripts/distill/runs/fulldata_v2/beyond4/score_clip_ll.py UNCHANGED, SCORE_STEP=4, ONE
# call per clip (all rows of a clip in the same call).  Changes vs the judge's script: GPU 0 under /tmp/claude-gpu0.lock,
# output files in THIS lane dir with a NEW tag, AYS5 render path resolved per MODE_S1.txt (reuse: the judge's renders for
# the 4 regime clips 0052 0147 0204 0301, this lane's for the other 8; own: this lane's for all 12), rows per STAGE, and
# no pairing side-effect (pairing is checked by check_pairing_ays5_v1.py).  Never overwrites.
# usage: score_ays5_v1.sh TAG STAGE CLIP [CLIP ...]      STAGE = S2 | S3
#   S2 rows: origin_ll, AYS5pad8_origin_g100, deliv_g100_T5pad, origin_g100_T5pad, deliv_g100_T5nat
#   S3 rows: deliv_g100_T5pad, AYS8_origin_g100 (this lane), AYS8_origin_g101 (judge),
#            + on the 4 regime clips: AYS5pad8_deliv_g100 (this lane), AYS5pad8_origin_g100 (resolved as in S2)
set -u
cd /home/kawa/master_project/StereoCrafter
PY=/home/kawa/miniconda3/envs/stereocrafter/bin/python
L=scripts/distill/runs/ays_20261004/ays5
O=outputs/ays_20261004/ays5/clips
JA=outputs/more_20261004/judge/ays/clips
F=outputs/finalcheck_20261004/speed/clips
B=outputs/beyond4_lossless/clips
TAG=$1; STAGE=$2; shift 2
MODE=$(cat $L/MODE_S1.txt 2>/dev/null | head -1)
case "$MODE" in reuse|own) ;; *) echo "BAD/MISSING MODE_S1.txt ($MODE)"; exit 2;; esac
ays5_path() {  # $1 = clip
  if [ "$MODE" = reuse ] && [[ " 0052 0147 0204 0301 " == *" $1 "* ]]; then
    echo $JA/${1}_AYS5pad8_origin_g100/${1}_inpainting_results_sbs.mkv
  else
    echo $O/${1}_AYS5pad8_origin_g100/${1}_inpainting_results_sbs.mkv
  fi
}
for C in "$@"; do
  OUTF=$L/SCORES_AYS5_${TAG}_$C.txt
  A5=$(ays5_path $C)
  case "$STAGE" in
    S2) SPECS="$C=$B/${C}_origin_ll/${C}_inpainting_results_sbs.mkv $C=$A5 $C=$F/${C}_deliv_g100_T5pad/${C}_inpainting_results_sbs.mkv $C=$F/${C}_origin_g100_T5pad/${C}_inpainting_results_sbs.mkv $C=$F/${C}_deliv_g100_T5nat/${C}_inpainting_results_sbs.mkv" ;;
    S3) SPECS="$C=$F/${C}_deliv_g100_T5pad/${C}_inpainting_results_sbs.mkv $C=$O/${C}_AYS8_origin_g100/${C}_inpainting_results_sbs.mkv $C=$JA/${C}_AYS8_origin_g101/${C}_inpainting_results_sbs.mkv"
        if [[ " 0052 0147 0204 0301 " == *" $C "* ]]; then
          SPECS="$SPECS $C=$O/${C}_AYS5pad8_deliv_g100/${C}_inpainting_results_sbs.mkv $C=$A5"
        fi ;;
    *) echo "BAD STAGE $STAGE"; exit 2 ;;
  esac
  for s in $SPECS; do [ -f "${s#*=}" ] || { echo "MISSING ${s#*=}"; exit 4; }; done
  [ -e "$OUTF" ] && { echo "refusing to overwrite $OUTF"; continue; }
  echo "SCORE_START $(date +%F_%T) scorer=scripts/distill/runs/fulldata_v2/beyond4/score_clip_ll.py SCORE_STEP=4 stage=$STAGE mode=$MODE" > $OUTF
  CUDA_VISIBLE_DEVICES=0 SCORE_STEP=4 flock /tmp/claude-gpu0.lock $PY scripts/distill/runs/fulldata_v2/beyond4/score_clip_ll.py $SPECS >> $OUTF 2>&1
  echo "SCORE_DONE rc=$? $(date +%F_%T)" >> $OUTF
  echo "SCORED $C stage=$STAGE -> $OUTF $(date +%H:%M:%S)"
done
