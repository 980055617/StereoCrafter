#!/bin/bash
# scale_gt ONE-TIME TEST of the dev-selected MAIN checkpoint (PREREG.txt): 12 test clips at 8 steps and at 25 steps
# (guidance 1.01, seed 1234, FFV1), UNREG scores, rows (+ published origin_ll / s25_ll rows), registered scores, analysis.
# usage: chain_test_v1.sh <STEP>       (checkpoint /mnt/ssd_data/deep_20261004/scale_gt/ck/main_v1/step<STEP>.pt)
set -u
cd /home/kawa/master_project/StereoCrafter
set +u; source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate stereocrafter; set -u
L=scripts/distill/runs/deep_20261004/scale_gt
S=$1
CK=/mnt/ssd_data/deep_20261004/scale_gt/ck/main_v1/step$S.pt
[ -f $CK ] || { echo "no checkpoint $CK"; exit 1; }
export CUDA_VISIBLE_DEVICES=1
TEST="0042 0052 0125 0128 0141 0147 0170 0204 0225 0251 0259 0301"
L8=selMain${S}_s8; L25=selMain${S}_s25
echo "TEST_START step=$S $(date +%F_%T)"
J=$L/jobs_test_main_v1_s${S}.txt; : > $J
for c in $TEST; do echo "$c $L8 $CK 8" >> $J; done
for c in $TEST; do echo "$c $L25 $CK 25" >> $J; done
bash $L/run_render_v1.sh $J
for LAB in $L8 $L25; do
  ARGS=""; for c in $TEST; do ARGS="$ARGS $c=outputs/deep_20261004/scale_gt/renders/${c}_${LAB}/${c}_inpainting_results_sbs.mkv"; done
  SCORE_STEP=4 flock /tmp/claude-gpu1.lock python scripts/distill/runs/fulldata_v2/beyond4/score_clip_ll.py $ARGS > $L/score_test_${LAB}.txt 2>&1 < /dev/null
  echo "SCORED $LAB: $(grep -c '^ROW' $L/score_test_${LAB}.txt) ROWs $(date +%F_%T)"
done
ROWS=$L/rows_test_main_v1_s${S}.json
python $L/make_rows_v1.py $ROWS origin_ll,s25_ll,$L8,$L25 $L/score_test_${L8}.txt $L/score_test_${L25}.txt \
  --base=scripts/distill/runs/more_20261004/eval_robustness/PUBLISHED_ROWS.json:origin_ll,s25_ll
O=outputs/deep_20261004/scale_gt/score_reg_test_main_v1_s${S}
[ -e $O ] && { echo "REFUSING: $O exists"; exit 1; }
mkdir -p $O
for c in $TEST; do
  SCORE_STEP=4 flock /tmp/claude-gpu1.lock python $L/score_registered_scalegt_v1.py $O $ROWS $c >> $L/score_reg_test_main_v1_s${S}.log 2>&1 < /dev/null
  echo "REGSCORE $c rc=$? $(date +%F_%T)"
done
SEL8=$L8 SEL25=$L25 python $L/analyze_v1.py test $O $L/TABLE_TEST_main_v1_s${S}.txt $ROWS
echo "TEST_DONE $(date +%F_%T)"
