#!/bin/bash
# external_models r2 main chain (PREREG_r2.txt): render batch A (regime set) under ONE GPU-0 lock hold -> score A
# (UNREG, REGISTERED, NR; each step its own lock hold) -> render batch B (remaining 8 test clips) -> score B.
# usage: bash chain_r2.sh <tag> "<clips A>" "<clips B>"
set -u
cd /home/kawa/master_project/StereoCrafter
R=scripts/distill/runs/deep_20261004/external_models
O=outputs/deep_20261004/external_models
PY=/mnt/ssd_data/deep_20261004/external_models/venv/bin/python
TAG=$1; A=$2; B=$3
LOG=$R/chain_${TAG}_r2.log
REG=$O/score_reg_${TAG}_r2
NR=$O/nr_${TAG}_r2
echo "CHAIN_START $(date '+%F_%T') tag=$TAG A=[$A] B=[$B]" >> $LOG
nr() {
  for c in "$@"; do
    M=$O/clips/${c}_${TAG}_ll/${c}_inpainting_results_sbs.mkv
    [ -s "$M" ] || { echo "NR_SKIP $c (no render)" >> $LOG; continue; }
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/mnt/ssd_data/deep_20261004/external_models/pylib_iqa \
      TORCH_HOME=/mnt/ssd_data/deep_20261004/external_models/torch_home HF_HUB_OFFLINE=1 \
      flock /tmp/claude-gpu0.lock /home/kawa/miniconda3/envs/stereocrafter/bin/python $R/nr_metrics_r2.py "$NR" "$c" \
      origin_ll=outputs/beyond4_lossless/clips/${c}_origin_ll/${c}_inpainting_results_sbs.mkv \
      deliv=outputs/beyond_distil_mamba_scaled/clips/${c}_mstudent2_step800_deliv_ll/${c}_inpainting_results_sbs.mkv \
      ${TAG}=$M >> $R/nr_${TAG}_r2.log 2>&1
    echo "NR_RC $c rc=$? $(date '+%F_%T')" >> $LOG
  done
}
for batch in A B; do
  CL=$A; [ $batch = B ] && CL=$B
  [ -z "$CL" ] && continue
  CUDA_VISIBLE_DEVICES=0 flock /tmp/claude-gpu0.lock $PY $R/m2svid_infer_ll.py --tag $TAG --win 16 --decode_chunk 8 --clips $CL \
    >> $R/render_${TAG}_${batch}_r2.log 2>&1
  echo "RENDER_$batch rc=$? $(date '+%F_%T')" >> $LOG
  bash $R/score_chain_r2.sh ${TAG}_ll "$REG" $CL
  echo "SCORE_$batch done $(date '+%F_%T')" >> $LOG
  nr $CL
done
echo "CHAIN_END $(date '+%F_%T')" >> $LOG
