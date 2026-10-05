#!/bin/bash
# vae_20261005 / decoder_swap -- DEV chain (PREREG.txt section 3).  Stages (every GPU call holds its GPU's lock via run_gpu_v1.sh):
#   redec    <gpu> "<decs>" "<clips>"   re-decode the listed clips x {origin_cap, deliv_cap} with each decoder
#   gate                                D1: stock SBS md5 == capture md5 on all 12 dev cells -> GATE_D1_dev.txt
#   unreg    <gpu> "<clips>"            score_clip_ll (UNREG, SCORE_STEP 4) on the two STOCK rows per clip
#   rows                                rows_dev_v1.json (shared left half; all labels)
#   reg      <gpu> "<clips>"            registered scorer + aux scorer per clip
#   temporal <gpu> "<clips>"            score_temporal_ll (tLP, RAFT warp, seam) per clip, all 10 rows (origin rows first)
#   pair     "<clips>"                  decode-pair flicker (CPU) per clip, all 10 rows
set -u
cd /home/kawa/master_project/StereoCrafter
L=scripts/distill/runs/vae_20261005/decoder_swap
R=/mnt/ssd_data/vae_20261005/decoder_swap/redec_dev
O=outputs/vae_20261005/decoder_swap
DEV="0040 0082 0091 0184 0245 0268"
DECS="stock stock32 ftmse ftema cd"
LABS=""; for lat in origin_cap deliv_cap; do for d in $DECS; do LABS="$LABS,${lat}__${d}"; done; done; LABS=${LABS#,}
cpu_env() { set +u; source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate stereocrafter; set -u;
            export PYTHONPATH=/mnt/ssd_data/vae_20261005/decoder_swap/pylib TORCH_HOME=/mnt/ssd_data/vae_20261005/decoder_swap/torch_home; }
STAGE=$1; shift
case $STAGE in
  redec)
    GPU=$1; D=$2; CL=$3
    PAIRS=""; for c in $CL; do PAIRS="$PAIRS,$c:origin_cap,$c:deliv_cap"; done; PAIRS=${PAIRS#,}
    TAG=$(echo "$D" | tr ' ' '_')_$(echo "$CL" | tr ' ' '_')
    bash $L/run_gpu_v1.sh $GPU $L/redec_dev_${TAG}.log $L/redecode_swap_v1.py $R ${D// /,} $PAIRS
    echo "REDEC [$D] [$CL] rc=$? $(date +%F_%T)" ;;
  gate)
    cpu_env; python $L/check_md5_swap_v1.py $L/GATE_D1_dev.txt $R ${DEV// /,}
    echo "GATE rc=$? $(date +%F_%T)" ;;
  unreg)
    GPU=$1; CL=$2
    for c in $CL; do
      F=$L/dev_unreg_${c}.txt
      [ -e $F ] && { echo "REFUSING: $F exists"; exit 1; }
      ARGS=""; for lab in origin_cap__stock deliv_cap__stock; do ARGS="$ARGS $c=$R/${c}_${lab}/${c}_inpainting_results_sbs.mkv"; done
      SCORE_STEP=4 bash $L/run_gpu_v1.sh $GPU $F $L/score_clip_ll.py $ARGS
      echo "UNREG $c rc=$? $(date +%F_%T)"
    done ;;
  rows)
    cpu_env; python $L/make_rows_shared_v1.py $L/rows_dev_v1.json $R $LABS $L/dev_unreg_*.txt
    echo "ROWS rc=$? $(date +%F_%T)" ;;
  reg)
    GPU=$1; CL=$2
    mkdir -p $O/score_reg_dev_v1 $O/score_aux_dev_v1
    for c in $CL; do
      SCORE_STEP=4 bash $L/run_gpu_v1.sh $GPU $L/score_reg_dev_v1_gpu$GPU.log $L/score_registered_df_v1.py $O/score_reg_dev_v1 $L/rows_dev_v1.json $c
      echo "REG $c rc=$? $(date +%F_%T)"
      bash $L/run_gpu_v1.sh $GPU $L/score_aux_dev_v1_gpu$GPU.log $L/score_aux_v1.py $O/score_aux_dev_v1 $L/rows_dev_v1.json $O/score_reg_dev_v1 $c
      echo "AUX $c rc=$? $(date +%F_%T)"
    done ;;
  temporal)
    GPU=$1; CL=$2
    mkdir -p $O/temporal_dev_v1
    for c in $CL; do
      [ -e $O/temporal_dev_v1/$c.json ] && { echo "REFUSING: $O/temporal_dev_v1/$c.json exists"; exit 1; }
      TA=""; for lab in ${LABS//,/ }; do TA="$TA $c=$R/${c}_${lab}/${c}_inpainting_results_sbs.mkv"; done
      bash $L/run_gpu_v1.sh $GPU $L/temporal_dev_v1_gpu$GPU.log $L/score_temporal_ll.py $O/temporal_dev_v1/$c.json $TA
      echo "TEMPORAL $c rc=$? $(date +%F_%T)"
    done ;;
  pair)
    CL=$1; cpu_env
    mkdir -p $O/pairflicker_dev_v1
    for c in $CL; do
      PA=""; for lab in ${LABS//,/ }; do PA="$PA $lab=$R/${c}_${lab}/${c}_inpainting_results_sbs.mkv"; done
      python $L/score_pairflicker_v1.py $O/pairflicker_dev_v1/$c.json $c $PA >> $L/pairflicker_dev_v1.log 2>&1 < /dev/null
      echo "PAIR $c rc=$? $(date +%F_%T)"
    done ;;
  *) echo "unknown stage $STAGE"; exit 1 ;;
esac
