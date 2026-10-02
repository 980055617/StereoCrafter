#!/bin/bash
# cross-check chain (GPU1): 2-frame-window inference of e001 vs origin on clip 0301 (the trainer's own window regime)
set -u; cd /home/kawa/master_project/StereoCrafter
set +u; source ~/miniconda3/etc/profile.d/conda.sh; conda activate stereocrafter; set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=1
D=scripts/distill/runs/diag_trainer; CK=weights/GTfinetune_v2_originattn_control/MambaCrafter_20260925_143200/train_state_epoch000001.pt
run() { L=$1; SC=$2; ST=$3; shift 3; OD=$D/infer/0301_$L; mkdir -p $OD; [ -f $OD/0301_inpainting_results_sbs.mp4 ] && { echo "HAVE $L"; return 0; }
  echo "START $L $(date +%T)"; python $D/lens1_infer_scaled_cond.py $SC $ST 0301 $OD "$@" > $OD.log 2>&1; echo "EXIT $L $? $(date +%T)"; }
run e001_fc2_condx1 1.0 $CK 2 1
run origin_fc2_condx1 1.0 origin 2 1
run e001_fc2_condx0.18215 0.18215 $CK 2 1
echo XCHECK1_DONE
