#!/bin/bash
# after chain_gpu0 finishes: retry the two runs that OOMed (another agent's probe held 17 GB on GPU 0), then re-score all 0301 variants.
set -u; cd /home/kawa/master_project/StereoCrafter
set +u; source ~/miniconda3/etc/profile.d/conda.sh; conda activate stereocrafter; set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
D=scripts/distill/runs/diag_trainer; CK=weights/GTfinetune_v2_originattn_control/MambaCrafter_20260925_143200/train_state_epoch000001.pt
until grep -q CHAIN_DONE $D/chain_gpu0.log; do sleep 10; done
run() { L=$1; SC=$2; ST=$3; shift 3; OD=$D/infer/0301_$L; mkdir -p $OD
  for attempt in 1 2 3 4 5 6; do
    G=0; F0=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0); F1=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1); [ "$F1" -lt "$F0" ] && G=1
    echo "START $L attempt=$attempt gpu=$G used0=$F0 used1=$F1 $(date +%T)"
    CUDA_VISIBLE_DEVICES=$G python $D/lens1_infer_scaled_cond.py $SC $ST 0301 $OD "$@" > $OD.log 2>&1 && { echo "EXIT $L 0 $(date +%T)"; return 0; }
    echo "FAIL $L attempt=$attempt $(date +%T)"; sleep 90
  done; return 1; }
run origin_condx0.18215 0.18215 origin
run e001_fc2_condx1 1.0 $CK 2 1
ARGS="0301=outputs/fulldata_v2/clips/0301_origin/0301_inpainting_results_sbs.mp4 0301=outputs/fulldata_v2/clips/0301_originattn_e001/0301_inpainting_results_sbs.mp4"
for L in e001_condx0.18215 origin_condx0.18215 e001_fc2_condx1 e001_fc2_condx0.18215 origin_fc2_condx1; do F=$D/infer/0301_$L/0301_inpainting_results_sbs.mp4; [ -f $F ] && ARGS="$ARGS 0301=$F"; done
CUDA_VISIBLE_DEVICES=1 python scripts/distill/score_clip.py $ARGS 2>&1 | grep -viE "warning|setting up|loading model|self.load_state|/home/kawa" | tee $D/infer/lpips_0301_variants.txt
echo RETRY_DONE
