#!/bin/bash
# judge J3c: ONE score_clip_ll.py call (UNCHANGED, SCORE_STEP=4) on 0301: origin_ll, AYS8, origin_g100_T5pad, AYS5pad8,
# deliverable T5@1.00 pad (context).  GPU 1 under its lock.  Also the per-window RNG-pairing check (CPU).
set -u
cd /home/kawa/master_project/StereoCrafter
PY=/home/kawa/miniconda3/envs/stereocrafter/bin/python
J=scripts/distill/runs/more_20261004/judge
A=outputs/more_20261004/judge/ays/clips
F=outputs/finalcheck_20261004/speed/clips
OUTF=$J/SCORES_J3c_AYS_0301.txt
SPECS="0301=outputs/beyond4_lossless/clips/0301_origin_ll/0301_inpainting_results_sbs.mkv 0301=$A/0301_AYS8_origin_g101/0301_inpainting_results_sbs.mkv 0301=$F/0301_origin_g100_T5pad/0301_inpainting_results_sbs.mkv 0301=$A/0301_AYS5pad8_origin_g100/0301_inpainting_results_sbs.mkv 0301=$F/0301_deliv_g100_T5pad/0301_inpainting_results_sbs.mkv"
for s in $SPECS; do [ -f "${s#*=}" ] || { echo "MISSING ${s#*=}"; exit 4; }; done
[ -e "$OUTF" ] && { echo "refusing to overwrite $OUTF"; exit 3; }
echo "SCORE_START $(date +%F_%T) scorer=scripts/distill/runs/fulldata_v2/beyond4/score_clip_ll.py SCORE_STEP=4" > $OUTF
CUDA_VISIBLE_DEVICES=1 SCORE_STEP=4 flock /tmp/claude-gpu1.lock $PY scripts/distill/runs/fulldata_v2/beyond4/score_clip_ll.py $SPECS >> $OUTF 2>&1
echo "SCORE_DONE rc=$? $(date +%F_%T)" >> $OUTF
$PY - <<'PY' > $J/J3c_AYS_PAIRING.txt
import json
A="outputs/more_20261004/judge/ays/clips"; F="outputs/finalcheck_20261004/speed/clips"
def fp(d):
    j=json.load(open(f"{d}/speed_log.json")); return [w["init_md5"] for w in j["windows"]], j
for a,b in [(f"{A}/0301_AYS8_origin_g101", f"{F}/0301_origin_g101_s8"), (f"{A}/0301_AYS5pad8_origin_g100", f"{F}/0301_origin_g100_T5pad"),
            (f"{A}/0301_AC0_origin_g101_T8sig", f"{F}/0301_origin_g101_s8")]:
    fa,ja=fp(a); fb,jb=fp(b); same=sum(x==y for x,y in zip(fa,fb))
    sl = ja.get("schedule") or {}
    print(f"{'PASS' if same==len(fa)==len(fb) else 'FAIL'} {a} vs {b}: {same}/{len(fa)} windows identical; pad_draws_total={ja.get('pad_draws_total')} calls/win={ja.get('unet_calls_per_window')} batch={ja.get('batch_sizes')} schedule={json.dumps(sl)[:400]}")
PY
cat $J/J3c_AYS_PAIRING.txt
echo "J3C_SCORING_DONE $(date +%F_%T)"
