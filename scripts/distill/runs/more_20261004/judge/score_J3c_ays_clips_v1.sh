#!/bin/bash
# judge J3c extension scoring: one score_clip_ll.py call (UNCHANGED, SCORE_STEP=4) per clip given as arguments, rows
# origin_ll, AYS8, origin_g100_T5pad, AYS5pad8 (gating) + mstudent2_step800_deliv_ll, deliv_g100_T5pad, s25_ll (context).
# GPU 1 under its lock.  Never overwrites.  usage: score_J3c_ays_clips_v1.sh TAG CLIP [CLIP ...]
set -u
cd /home/kawa/master_project/StereoCrafter
PY=/home/kawa/miniconda3/envs/stereocrafter/bin/python
J=scripts/distill/runs/more_20261004/judge
A=outputs/more_20261004/judge/ays/clips
F=outputs/finalcheck_20261004/speed/clips
TAG=$1; shift
for C in "$@"; do
  OUTF=$J/SCORES_J3c_AYS_${TAG}_$C.txt
  S25=outputs/skeptic1_stack/clips/${C}_s25_ll/${C}_inpainting_results_sbs.mkv
  [ -f $S25 ] || S25=outputs/beyond4_lossless/clips/${C}_s25_ll/${C}_inpainting_results_sbs.mkv   # published s25 path (eval_robustness PUBLISHED_ROWS.json)
  SPECS="$C=outputs/beyond4_lossless/clips/${C}_origin_ll/${C}_inpainting_results_sbs.mkv $C=$A/${C}_AYS8_origin_g101/${C}_inpainting_results_sbs.mkv $C=$F/${C}_origin_g100_T5pad/${C}_inpainting_results_sbs.mkv"
  [ -f $A/${C}_AYS5pad8_origin_g100/${C}_inpainting_results_sbs.mkv ] && SPECS="$SPECS $C=$A/${C}_AYS5pad8_origin_g100/${C}_inpainting_results_sbs.mkv"
  SPECS="$SPECS $C=outputs/beyond_distil_mamba_scaled/clips/${C}_mstudent2_step800_deliv_ll/${C}_inpainting_results_sbs.mkv $C=$F/${C}_deliv_g100_T5pad/${C}_inpainting_results_sbs.mkv $C=$S25"
  for s in $SPECS; do [ -f "${s#*=}" ] || { echo "MISSING ${s#*=}"; exit 4; }; done
  [ -e "$OUTF" ] && { echo "refusing to overwrite $OUTF"; continue; }
  echo "SCORE_START $(date +%F_%T) scorer=scripts/distill/runs/fulldata_v2/beyond4/score_clip_ll.py SCORE_STEP=4" > $OUTF
  CUDA_VISIBLE_DEVICES=1 SCORE_STEP=4 flock /tmp/claude-gpu1.lock $PY scripts/distill/runs/fulldata_v2/beyond4/score_clip_ll.py $SPECS >> $OUTF 2>&1
  echo "SCORE_DONE rc=$? $(date +%F_%T)" >> $OUTF
done
$PY - "$@" <<'PY' >> $J/J3c_AYS_PAIRING_EXT.txt
import json, os, sys
A="outputs/more_20261004/judge/ays/clips"; F="outputs/finalcheck_20261004/speed/clips"
def fp(d):
    j=json.load(open(f"{d}/speed_log.json")); return [w["init_md5"] for w in j["windows"]]
for c in sys.argv[1:]:
    ref8 = f"{F}/{c}_origin_g101_s8" if os.path.exists(f"{F}/{c}_origin_g101_s8/speed_log.json") else f"{F}/{c}_origin_g100_s8"
    pairs = [(f"{A}/{c}_AYS8_origin_g101", ref8)]
    if os.path.exists(f"{A}/{c}_AYS5pad8_origin_g100/speed_log.json"): pairs.append((f"{A}/{c}_AYS5pad8_origin_g100", f"{F}/{c}_origin_g100_T5pad"))
    for a, b in pairs:
        fa, fb = fp(a), fp(b); same = sum(x == y for x, y in zip(fa, fb))
        print(f"{'PASS' if same == len(fa) == len(fb) else 'FAIL'} {a} vs {b}: {same}/{len(fa)} windows identical")
PY
tail -n 8 $J/J3c_AYS_PAIRING_EXT.txt
echo "J3C_EXT_SCORING_DONE $(date +%F_%T)"
