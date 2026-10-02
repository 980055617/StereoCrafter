#!/bin/bash
# GPU-1 chain: 2-frame-window inference (the trainer's stage-3 regime) on clip 0042 for origin and e001, then LPIPS vs GT.
set -u; cd /home/kawa/master_project/StereoCrafter
set +u; source ~/miniconda3/etc/profile.d/conda.sh; conda activate stereocrafter; set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=1
D=scripts/distill/runs/diag_trainer; CK=weights/GTfinetune_v2_originattn_control/MambaCrafter_20260925_143200/train_state_epoch000001.pt
run() { L=$1; SC=$2; ST=$3; shift 3; OD=$D/infer/0042_$L; mkdir -p $OD; echo "START $L $(date +%T)"; python $D/lens1_infer_scaled_cond.py $SC $ST 0042 $OD "$@" > $OD.log 2>&1; echo "EXIT $L $? $(date +%T)"; }
run e001_fc2_condx1 1.0 $CK 2 1
run origin_fc2_condx1 1.0 origin 2 1
ARGS="0042=outputs/fulldata_v2/clips/0042_origin/0042_inpainting_results_sbs.mp4 0042=outputs/fulldata_v2/clips/0042_originattn_e001/0042_inpainting_results_sbs.mp4"
for L in e001_fc2_condx1 origin_fc2_condx1; do ARGS="$ARGS 0042=$D/infer/0042_$L/0042_inpainting_results_sbs.mp4"; done
python scripts/distill/score_clip.py $ARGS 2>&1 | grep -viE "warning|setting up|loading model|self.load_state|/home/kawa" | tee $D/infer/lpips_0042_variants.txt
echo CHAIN1_DONE
