#!/bin/bash
# decoder_ft: DEV stock stage.  (1) stock re-decode of the 6 dev clips x 2 models (GPU, one lock hold) -> C1/C2 gates;
# (2) the unsharp-baseline grid rows (CPU, 6 in parallel) from the stock re-decodes.  No scoring here.
# usage: bash chain_dev_stock_v1.sh <gpu>
set -u
cd /home/kawa/master_project/StereoCrafter
set +u; source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate stereocrafter; set -u
L=scripts/distill/runs/deep_20261004/decoder_ft
GPU=$1
R=/mnt/ssd_data/deep_20261004/decoder_ft/redec_dev
DEV="0040 0082 0091 0184 0245 0268"
PAIRS=""
for c in $DEV; do PAIRS="$PAIRS,$c:origin_cap,$c:deliv_cap"; done
PAIRS=${PAIRS#,}
echo "STOCK_START $(date +%F_%T)"
CUDA_VISIBLE_DEVICES=$GPU flock /tmp/claude-gpu$GPU.lock python $L/redecode_v1.py $R stock $PAIRS > $L/redec_dev_stock_v1.log 2>&1 < /dev/null
echo "STOCK_REDECODE rc=$? $(date +%F_%T)"
python $L/check_md5_v1.py $L/GATES_C1C2_dev.txt $R ${DEV// /,}
echo "UNSHARP_START $(date +%F_%T)"
N=0
for c in $DEV; do
  for lat in origin_cap deliv_cap; do
    for s in 1 2; do
      for a in 0.15 0.30 0.50; do
        python $L/make_unsharp_v1.py $R/${c}_${lat}__stock/${c}_inpainting_results_sbs.mkv $R ${c}_${lat}__us_s${s}a${a} $s $a >> $L/unsharp_dev_v1.log 2>&1 &
        N=$((N+1))
        if [ $((N % 6)) -eq 0 ]; then wait; fi
      done
    done
  done
done
wait
echo "DEV_STOCK_DONE $(date +%F_%T)"
