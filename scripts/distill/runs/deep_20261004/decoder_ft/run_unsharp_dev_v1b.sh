#!/bin/bash
# decoder_ft: re-run of the DEV unsharp grid (chain_dev_stock_v1.sh's unsharp step segfaulted at import in every process:
# cv2 imported before torch/diffusers; make_unsharp_v1.py fixed to import the FFV1 writer first; no row had been written).
set -u
cd /home/kawa/master_project/StereoCrafter
set +u; source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate stereocrafter; set -u
L=scripts/distill/runs/deep_20261004/decoder_ft
R=/mnt/ssd_data/deep_20261004/decoder_ft/redec_dev
N=0
for c in 0040 0082 0091 0184 0245 0268; do for lat in origin_cap deliv_cap; do for s in 1 2; do for a in 0.15 0.30 0.50; do
  python $L/make_unsharp_v1.py $R/${c}_${lat}__stock/${c}_inpainting_results_sbs.mkv $R ${c}_${lat}__us_s${s}a${a} $s $a >> $L/unsharp_dev_v1b.log 2>&1 &
  N=$((N+1)); if [ $((N % 6)) -eq 0 ]; then wait; fi
done; done; done; done
wait
echo "UNSHARP_DEV_DONE $(ls -d $R/*__us_* | wc -l) rows $(date +%F_%T)"
