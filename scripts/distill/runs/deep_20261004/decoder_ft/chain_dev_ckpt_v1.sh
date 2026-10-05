#!/bin/bash
# decoder_ft DEV checkpoint stage (PREREG.txt section 4), split into separately runnable parts so the two GPUs can share it:
#   redec   <gpu> <run> "<steps>"            re-decode the 6 dev clips x 2 models with each listed checkpoint
#   c3      <gpu> <run>                      step-0 control on 0040 (both models) + GATES_C3 file
#   score   <gpu> <run> "<steps>" "<clips>"  score_clip_ll on the two STOCK rows of the listed clips (every row shares
#                                            the left half; UNREG of every row comes from the registered scorer)
#   reg     <gpu> <run> "<steps>" "<clips>"  rows json (built once, all 6 clips) + registered + aux scorers for the listed clips
#   analyze <run> "<steps>"                  analyze_dev_v1.py -> TABLE_DEV_<run>.txt, SELECTION_DEV_<run>.json
# Every GPU call holds the lock of the GPU it uses.
set -u
cd /home/kawa/master_project/StereoCrafter
set +u; source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate stereocrafter; set -u
L=scripts/distill/runs/deep_20261004/decoder_ft
R=/mnt/ssd_data/deep_20261004/decoder_ft/redec_dev
DEV="0040 0082 0091 0184 0245 0268"
STAGE=$1; shift
labs() {   # all dev labels for the given steps
  local NAMES="stock"; for s in $1; do NAMES="$NAMES s$s"; done
  for s in 1 2; do for a in 0.15 0.30 0.50; do NAMES="$NAMES us_s${s}a${a}"; done; done
  local OUT=""; for lat in origin_cap deliv_cap; do for nm in $NAMES; do OUT="$OUT,${lat}__${nm}"; done; done; echo ${OUT#,}
}
case $STAGE in
  redec)
    GPU=$1; RUN=$2; STEPS=$3; CK=/mnt/ssd_data/deep_20261004/decoder_ft/ck/$RUN
    PAIRS=""; for c in $DEV; do PAIRS="$PAIRS,$c:origin_cap,$c:deliv_cap"; done; PAIRS=${PAIRS#,}
    SPEC=""; for s in $STEPS; do SPEC="$SPEC,s$s=$CK/step$s.pt"; done; SPEC=${SPEC#,}
    TAG=$(echo $STEPS | tr ' ' '_')
    CUDA_VISIBLE_DEVICES=$GPU flock /tmp/claude-gpu$GPU.lock python $L/redecode_v1.py $R $SPEC $PAIRS > $L/redec_dev_${RUN}_s${TAG}.log 2>&1 < /dev/null
    echo "REDECODE steps=[$STEPS] rc=$? $(date +%F_%T)" ;;
  c3)
    GPU=$1; RUN=$2; CK=/mnt/ssd_data/deep_20261004/decoder_ft/ck/$RUN
    CUDA_VISIBLE_DEVICES=$GPU flock /tmp/claude-gpu$GPU.lock python $L/redecode_v1.py $R s0=$CK/step0.pt 0040:origin_cap,0040:deliv_cap > $L/redec_dev_${RUN}_s0.log 2>&1 < /dev/null
    python $L/check_md5_v1.py $L/GATES_C3_dev_${RUN}.txt $R 0040 s0
    echo "C3 done $(date +%F_%T)" ;;
  score)
    GPU=$1; RUN=$2; STEPS=$3; CL=$4; LABS=$(labs "$STEPS")
    for c in $CL; do
      F=$L/dev_unreg_${c}_${RUN}.txt
      [ -e $F ] && { echo "REFUSING: $F exists"; exit 1; }
      ARGS=""; for lab in origin_cap__stock deliv_cap__stock; do ARGS="$ARGS $c=$R/${c}_${lab}/${c}_inpainting_results_sbs.mkv"; done   # stock rows only (make_rows_shared_v1.py)
      SCORE_STEP=4 CUDA_VISIBLE_DEVICES=$GPU flock /tmp/claude-gpu$GPU.lock python scripts/distill/runs/fulldata_v2/beyond4/score_clip_ll.py $ARGS > $F 2>&1 < /dev/null
      echo "UNREG $c rc=$? $(date +%F_%T)"
    done ;;
  reg)
    GPU=$1; RUN=$2; STEPS=$3; CL=$4; LABS=$(labs "$STEPS")
    ROWS=$L/rows_dev_${RUN}.json
    O=outputs/deep_20261004/decoder_ft/score_reg_dev_${RUN}
    A=outputs/deep_20261004/decoder_ft/score_aux_dev_${RUN}
    mkdir -p $O $A
    ( flock 9; [ -e $ROWS ] || python $L/make_rows_shared_v1.py $ROWS $R $LABS $L/dev_unreg_*_${RUN}.txt ) 9>$L/.rows_dev_${RUN}.lock || exit 1
    for c in $CL; do
      SCORE_STEP=4 CUDA_VISIBLE_DEVICES=$GPU flock /tmp/claude-gpu$GPU.lock python $L/score_registered_df_v1.py $O $ROWS $c >> $L/score_reg_dev_${RUN}_gpu$GPU.log 2>&1 < /dev/null
      echo "REG $c rc=$? $(date +%F_%T)"
      CUDA_VISIBLE_DEVICES=$GPU flock /tmp/claude-gpu$GPU.lock env PYTHONPATH=/mnt/ssd_data/deep_20261004/blur_diag/pylib:/mnt/ssd_data/deep_20261004/skeptic/pylib \
        TORCH_HOME=/mnt/ssd_data/deep_20261004/decoder_ft/torch_home HF_HOME=/mnt/ssd_data/deep_20261004/blur_diag/hf_home \
        HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python $L/score_aux_v1.py $A $ROWS $O $c >> $L/score_aux_dev_${RUN}_gpu$GPU.log 2>&1 < /dev/null
      echo "AUX $c rc=$? $(date +%F_%T)"
    done ;;
  analyze)
    RUN=$1; STEPS=$2
    python $L/analyze_dev_v1.py $RUN "$STEPS" > $L/TABLE_DEV_${RUN}.txt 2>&1
    echo "ANALYZE rc=$? $(date +%F_%T)" ;;
  *) echo "unknown stage $STAGE"; exit 1 ;;
esac
