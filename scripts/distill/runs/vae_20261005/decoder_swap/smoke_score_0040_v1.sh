#!/bin/bash
# vae_20261005 / decoder_swap -- one-clip SMOKE of the whole dev scoring chain on 0040 (separate smoke outputs; the dev run
# re-scores 0040 into its own dirs).  GPU <gpu> under its lock.
set -u
GPU=$1
cd /home/kawa/master_project/StereoCrafter
L=scripts/distill/runs/vae_20261005/decoder_swap
R=/mnt/ssd_data/vae_20261005/decoder_swap/redec_dev
O=outputs/vae_20261005/decoder_swap/smoke_0040
mkdir -p $O
set +u; source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate stereocrafter; set -u
export PYTHONPATH=/mnt/ssd_data/vae_20261005/decoder_swap/pylib TORCH_HOME=/mnt/ssd_data/vae_20261005/decoder_swap/torch_home
LABS=""; for lat in origin_cap deliv_cap; do for d in stock stock32 ftmse ftema cd; do LABS="$LABS,${lat}__${d}"; done; done; LABS=${LABS#,}
ARGS=""; for lab in origin_cap__stock deliv_cap__stock; do ARGS="$ARGS 0040=$R/0040_${lab}/0040_inpainting_results_sbs.mkv"; done
SCORE_STEP=4 bash $L/run_gpu_v1.sh $GPU $L/smoke_unreg_0040.txt $L/score_clip_ll.py $ARGS
python $L/make_rows_shared_v1.py $L/rows_smoke_0040.json $R $LABS $L/smoke_unreg_0040.txt
SCORE_STEP=4 bash $L/run_gpu_v1.sh $GPU $L/smoke_score_0040.log $L/score_registered_df_v1.py $O/reg $L/rows_smoke_0040.json 0040
bash $L/run_gpu_v1.sh $GPU $L/smoke_score_0040.log $L/score_aux_v1.py $O/aux $L/rows_smoke_0040.json $O/reg 0040
TA=""; for lab in ${LABS//,/ }; do TA="$TA 0040=$R/0040_${lab}/0040_inpainting_results_sbs.mkv"; done
bash $L/run_gpu_v1.sh $GPU $L/smoke_score_0040.log $L/score_temporal_ll.py $O/temporal_0040.json $TA
PA=""; for lab in ${LABS//,/ }; do PA="$PA $lab=$R/0040_${lab}/0040_inpainting_results_sbs.mkv"; done
python $L/score_pairflicker_v1.py $O/pairflicker_0040.json 0040 $PA >> $L/smoke_score_0040.log 2>&1
echo "SMOKE_SCORE_DONE $(date +%F_%T)" >> $L/smoke_score_0040.log
