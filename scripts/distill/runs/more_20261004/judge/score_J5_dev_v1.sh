#!/bin/bash
# judge J5: one score_clip_ll.py call (UNCHANGED, SCORE_STEP=4) per dev clip with the five rows; GPU 1 under its lock.
# Then the per-window RNG pairing check (CPU).  Never overwrites.
set -u
cd /home/kawa/master_project/StereoCrafter
PY=/home/kawa/miniconda3/envs/stereocrafter/bin/python
J=scripts/distill/runs/more_20261004/judge
D=outputs/more_20261004/judge/devclips/clips
for C in 0268 0082; do
  OUTF=$J/SCORES_J5_DEV_$C.txt
  SPECS=""
  for L in o8_origin_g101_s8 d8_deliv_g101_s8 o5p_origin_g100_T5pad d5p_deliv_g100_T5pad d5n_deliv_g100_T5nat; do
    SPECS="$SPECS $C=$D/${C}_$L/${C}_inpainting_results_sbs.mkv"
  done
  for s in $SPECS; do [ -f "${s#*=}" ] || { echo "MISSING ${s#*=}"; exit 4; }; done
  [ -e "$OUTF" ] && { echo "refusing to overwrite $OUTF"; continue; }
  echo "SCORE_START $(date +%F_%T) scorer=scripts/distill/runs/fulldata_v2/beyond4/score_clip_ll.py SCORE_STEP=4" > $OUTF
  CUDA_VISIBLE_DEVICES=1 SCORE_STEP=4 flock /tmp/claude-gpu1.lock $PY scripts/distill/runs/fulldata_v2/beyond4/score_clip_ll.py $SPECS >> $OUTF 2>&1
  echo "SCORE_DONE rc=$? $(date +%F_%T)" >> $OUTF
done
$PY - <<'PY' > $J/J5_DEV_PAIRING.txt
import json
D="outputs/more_20261004/judge/devclips/clips"
for c in ["0268","0082"]:
    ref=[w["init_md5"] for w in json.load(open(f"{D}/{c}_o8_origin_g101_s8/speed_log.json"))["windows"]]
    for L in ["d8_deliv_g101_s8","o5p_origin_g100_T5pad","d5p_deliv_g100_T5pad","d5n_deliv_g100_T5nat"]:
        j=json.load(open(f"{D}/{c}_{L}/speed_log.json")); f=[w["init_md5"] for w in j["windows"]]
        same=sum(a==b for a,b in zip(f,ref)); full=(same==len(f)==len(ref))
        exp = "window 0 only (unpadded)" if L.endswith("T5nat") else "all windows"
        print(f"{c} {L} vs o8: {same}/{len(f)} windows identical (expected {exp}) -> {'PASS' if (full or (L.endswith('T5nat') and f[0]==ref[0])) else 'FAIL'} pad_draws_total={j.get('pad_draws_total')} calls/win={j.get('unet_calls_per_window')} batch={j.get('batch_sizes')}")
PY
cat $J/J5_DEV_PAIRING.txt
echo "J5_SCORING_DONE $(date +%F_%T)"
