#!/bin/bash
# sequential GPU-0 chain: waits for GPU 0 to be free, then runs the remaining inference variants on clip 0301 and scores them.
set -u; cd /home/kawa/master_project/StereoCrafter
set +u; source ~/miniconda3/etc/profile.d/conda.sh; conda activate stereocrafter; set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0
D=scripts/distill/runs/diag_trainer; CK=weights/GTfinetune_v2_originattn_control/MambaCrafter_20260925_143200/train_state_epoch000001.pt
# (wait removed: snapd-desktop-integration always holds GPU 0)
run() { L=$1; SC=$2; ST=$3; shift 3; OD=$D/infer/0301_$L; mkdir -p $OD; echo "START $L $(date +%T)"; python $D/lens1_infer_scaled_cond.py $SC $ST 0301 $OD "$@" > $OD.log 2>&1; echo "EXIT $L $? $(date +%T)"; }
run origin_condx0.18215 0.18215 origin
run e001_fc2_condx1 1.0 $CK 2 1
run e001_fc2_condx0.18215 0.18215 $CK 2 1
run origin_fc2_condx1 1.0 origin 2 1
ARGS="0301=outputs/fulldata_v2/clips/0301_origin/0301_inpainting_results_sbs.mp4 0301=outputs/fulldata_v2/clips/0301_originattn_e001/0301_inpainting_results_sbs.mp4"
for L in e001_condx0.18215 origin_condx0.18215 e001_fc2_condx1 e001_fc2_condx0.18215 origin_fc2_condx1; do ARGS="$ARGS 0301=$D/infer/0301_$L/0301_inpainting_results_sbs.mp4"; done
python scripts/distill/score_clip.py $ARGS 2>&1 | grep -viE "warning|setting up|loading model|self.load_state|/home/kawa" | tee $D/infer/lpips_0301_variants.txt
echo CHAIN_DONE
