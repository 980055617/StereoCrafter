#!/bin/bash
# after chain_gpu0 finishes: (re)run every 0301 variant whose output is missing (GPU 0 was shared with another agent's 17 GB probe), then score.
set -u; cd /home/kawa/master_project/StereoCrafter
set +u; source ~/miniconda3/etc/profile.d/conda.sh; conda activate stereocrafter; set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
D=scripts/distill/runs/diag_trainer; CK=weights/GTfinetune_v2_originattn_control/MambaCrafter_20260925_143200/train_state_epoch000001.pt
until grep -q CHAIN_DONE $D/chain_gpu0.log; do sleep 10; done
until grep -q CHAIN1_DONE $D/chain_gpu1.log; do sleep 10; done   # one job of ours per GPU at a time
pick_gpu() { F0=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0); F1=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1); G=0; [ "$F1" -lt "$F0" ] && G=1; echo "$G $F0 $F1"; }
run() { L=$1; SC=$2; ST=$3; shift 3; OD=$D/infer/0301_$L; mkdir -p $OD; [ -f $OD/0301_inpainting_results_sbs.mp4 ] && { echo "HAVE $L"; return 0; }
  for attempt in 1 2 3 4 5 6 7 8; do
    read G F0 F1 < <(pick_gpu)
    echo "START $L attempt=$attempt gpu=$G used0=$F0 used1=$F1 $(date +%T)"
    CUDA_VISIBLE_DEVICES=$G python $D/lens1_infer_scaled_cond.py $SC $ST 0301 $OD "$@" > $OD.log 2>&1 && { echo "EXIT $L 0 $(date +%T)"; return 0; }
    echo "FAIL $L attempt=$attempt $(date +%T)"; sleep 90
  done; return 1; }
run origin_condx0.18215 0.18215 origin
run e001_fc2_condx1 1.0 $CK 2 1
run e001_fc2_condx0.18215 0.18215 $CK 2 1
run origin_fc2_condx1 1.0 origin 2 1
ARGS="0301=outputs/fulldata_v2/clips/0301_origin/0301_inpainting_results_sbs.mp4 0301=outputs/fulldata_v2/clips/0301_originattn_e001/0301_inpainting_results_sbs.mp4"
for L in e001_condx0.18215 origin_condx0.18215 e001_fc2_condx1 e001_fc2_condx0.18215 origin_fc2_condx1; do F=$D/infer/0301_$L/0301_inpainting_results_sbs.mp4; [ -f $F ] && ARGS="$ARGS 0301=$F"; done
read G F0 F1 < <(pick_gpu)
CUDA_VISIBLE_DEVICES=$G python scripts/distill/score_clip.py $ARGS 2>&1 | grep -viE "warning|setting up|loading model|self.load_state|/home/kawa" | tee $D/infer/lpips_0301_variants.txt
echo RETRY_DONE
