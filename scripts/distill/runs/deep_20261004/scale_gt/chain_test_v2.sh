#!/bin/bash
# scale_gt ONE-TIME TEST of the dev-selected MAIN checkpoint (PREREG.txt + PREREG_ADDENDUM_2.txt).
# v2 = chain_test_v1.sh + the colour-removed sensitivity rows (color_remove_v1.py, reference = origin_ll / s25_ll, never the GT).
#   renders: 12 test clips at 8 steps and at 25 steps (guidance 1.01, seed 1234, FFV1), each its own GPU-1 lock
#   UNREG scores (score_clip_ll.py) -> colour-removed copies (CPU) -> their UNREG scores -> rows json (+ published origin_ll/s25_ll)
#   -> registered scores (one call per clip, all 6 labels) -> analysis table.
# usage: chain_test_v2.sh <STEP>       (checkpoint /mnt/ssd_data/deep_20261004/scale_gt/ck/main_v1/step<STEP>.pt)
set -u
cd /home/kawa/master_project/StereoCrafter
set +u; source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate stereocrafter; set -u
L=scripts/distill/runs/deep_20261004/scale_gt
S=$1
CK=/mnt/ssd_data/deep_20261004/scale_gt/ck/main_v1/step$S.pt
[ -f $CK ] || { echo "no checkpoint $CK"; exit 1; }
export CUDA_VISIBLE_DEVICES=1
TEST="0042 0052 0125 0128 0141 0147 0170 0204 0225 0251 0259 0301"
R=outputs/deep_20261004/scale_gt/renders
L8=selMain${S}_s8; L25=selMain${S}_s25
echo "TEST_START step=$S $(date +%F_%T)"
J=$L/jobs_test_main_v1_s${S}.txt
if [ ! -e $J ]; then
  for c in $TEST; do echo "$c $L8 $CK 8" >> $J; done
  for c in $TEST; do echo "$c $L25 $CK 25" >> $J; done
fi
bash $L/run_render_v1.sh $J
for LAB in $L8 $L25; do
  F=$L/score_test_${LAB}.txt
  if [ ! -e $F ]; then
    ARGS=""; for c in $TEST; do ARGS="$ARGS $c=$R/${c}_${LAB}/${c}_inpainting_results_sbs.mkv"; done
    SCORE_STEP=4 flock /tmp/claude-gpu1.lock python scripts/distill/runs/fulldata_v2/beyond4/score_clip_ll.py $ARGS > $F 2>&1 < /dev/null
  fi
  echo "SCORED $LAB: $(grep -c '^ROW' $F) ROWs $(date +%F_%T)"
done
# colour-removed copies (CPU only); references = the published origin_ll (8 steps) / s25_ll (25 steps) renders
O=outputs/deep_20261004/scale_gt/diag_colorremoved_test_v1
mkdir -p $O
CR8=""; CR25=""
for c in $TEST; do
  REF8=$(python -c "import json;print(json.load(open('scripts/distill/runs/more_20261004/eval_robustness/PUBLISHED_ROWS.json'))['cells']['$c']['origin_ll']['path'])")
  REF25=$(python -c "import json;print(json.load(open('scripts/distill/runs/more_20261004/eval_robustness/PUBLISHED_ROWS.json'))['cells']['$c']['s25_ll']['path'])")
  [ -e $O/${c}_${L8}_colrm ] || CR8="$CR8 $c:$R/${c}_${L8}/${c}_inpainting_results_sbs.mkv:$REF8"
  [ -e $O/${c}_${L25}_colrm ] || CR25="$CR25 $c:$R/${c}_${L25}/${c}_inpainting_results_sbs.mkv:$REF25"
done
[ -n "$CR8$CR25" ] && CUDA_VISIBLE_DEVICES='' python $L/color_remove_v1.py $O $CR8 $CR25 >> $L/color_remove_test_v1.log 2>&1
echo "COLRM done $(date +%F_%T)"
for LAB in $L8 $L25; do
  F=$L/score_test_${LAB}_colrm.txt
  if [ ! -e $F ]; then
    ARGS=""; for c in $TEST; do ARGS="$ARGS $c=$O/${c}_${LAB}_colrm/${c}_inpainting_results_sbs.mkv"; done
    SCORE_STEP=4 flock /tmp/claude-gpu1.lock python scripts/distill/runs/fulldata_v2/beyond4/score_clip_ll.py $ARGS > $F 2>&1 < /dev/null
  fi
  echo "SCORED ${LAB}_colrm: $(grep -c '^ROW' $F) ROWs $(date +%F_%T)"
done
ROWS=$L/rows_test_main_v1_s${S}.json
[ -e $ROWS ] || python $L/make_rows_v1.py $ROWS origin_ll,s25_ll,$L8,$L25,${L8}_colrm,${L25}_colrm \
  $L/score_test_${L8}.txt $L/score_test_${L25}.txt $L/score_test_${L8}_colrm.txt $L/score_test_${L25}_colrm.txt \
  --base=scripts/distill/runs/more_20261004/eval_robustness/PUBLISHED_ROWS.json:origin_ll,s25_ll || exit 1
OR=outputs/deep_20261004/scale_gt/score_reg_test_main_v1_s${S}
mkdir -p $OR
for c in $TEST; do
  [ -e $OR/$c.json ] && { echo "REGSCORE $c exists"; continue; }
  SCORE_STEP=4 flock /tmp/claude-gpu1.lock python $L/score_registered_scalegt_v1.py $OR $ROWS $c >> $L/score_reg_test_main_v1_s${S}.log 2>&1 < /dev/null
  echo "REGSCORE $c rc=$? $(date +%F_%T)"
done
SEL8=$L8 SEL25=$L25 python $L/analyze_v1.py test $OR $L/TABLE_TEST_main_v1_s${S}.txt $ROWS
echo "TEST_DONE $(date +%F_%T)"
