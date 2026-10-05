#!/bin/bash
# CPU scoring of batch B (8 remaining test clips) as each render lands (deviation D3, D4): UNREG (score_clip_ll_cpu_r2.py,
# origin re-scored), REGISTERED for M2SVid only (score_reg_m2only_r2.py SKIP_REF=1; origin / deliverable = published GPU
# values), temporal (temporal_cpu_r2.py).  NR only on the regime set.  At most 3 clips in flight.
set -u
cd /home/kawa/master_project/StereoCrafter
set +u; source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate stereocrafter; set -u
export CUDA_VISIBLE_DEVICES="" SCORE_STEP=4 NTHREADS=4
R=scripts/distill/runs/deep_20261004/external_models
O=outputs/deep_20261004/external_models
TAG=m2svid_fa_w16; LABEL=${TAG}_ll
one() {
  c=$1; M=$O/clips/${c}_${LABEL}/${c}_inpainting_results_sbs.mkv
  echo "B_START $c $(date '+%F_%T')" >> $R/score_cpu_r2.log
  python $R/score_clip_ll_cpu_r2.py "$c=outputs/beyond4_lossless/clips/${c}_origin_ll/${c}_inpainting_results_sbs.mkv" "$c=$M" \
    > $R/unreg_cpu/${c}.txt 2>&1 &
  SKIP_REF=1 python $R/score_reg_m2only_r2.py $O/score_reg_cpu_${TAG}_r2 $c ${LABEL}=$M > $R/reg_cpu/${c}.log 2>&1 &
  python $R/temporal_cpu_r2.py $O/temporal_cpu_${TAG}_r2 $TAG $c > $R/temporal_cpu/${c}.log 2>&1 &
  wait
  echo "B_END $c $(date '+%F_%T')" >> $R/score_cpu_r2.log
}
for c in "$@"; do
  until [ -e $O/clips/${c}_${LABEL}/run_${c}.json ]; do
    grep -q "RENDER_B" $R/chain_${TAG}_r2.log 2>/dev/null && [ ! -e $O/clips/${c}_${LABEL}/run_${c}.json ] && \
      { echo "B_NORENDER $c" >> $R/score_cpu_r2.log; continue 2; }
    sleep 10
  done
  while [ "$(jobs -rp | wc -l)" -ge 3 ]; do sleep 5; done
  one $c &
done
wait
echo "B_ALL_END $(date '+%F_%T')" >> $R/score_cpu_r2.log
