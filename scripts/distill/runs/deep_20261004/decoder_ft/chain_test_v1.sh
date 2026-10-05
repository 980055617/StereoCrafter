#!/bin/bash
# decoder_ft TEST stage (PREREG.txt section 5) -- ONE evaluation with the dev-selected decoder checkpoint SEL and the
# dev-selected unsharp baseline US.  Split into stages so the two GPUs can share it (each GPU call holds its GPU's lock):
#   redec    <gpu> <run> <sel> "<clips>"          stock + SEL re-decodes into outputs/deep_20261004/decoder_ft/test_renders
#   gates                                         C1/C2 md5 gates over all 12 clips
#   unsharp  <us> "<clips>"                       US rows (CPU) from the stock re-decodes
#   unreg    <gpu> <run> <sel> <us> "<clips>"     score_clip_ll UNREG rows
#   rows     <run> <sel> <us>                     rows json (+ published origin_ll / s25_ll / AYS8 / deliverable rows)
#   score    <gpu> <run> <sel> "<clips>"          registered + aux + temporal (GPU) + pair flicker (CPU) per clip
#   roundtrip <gpu> <run> <sel>                   decoder-only GT round trip on the 12 test clips (diagnostic)
#   panels   <run> <sel>                          g8 visual panels (0301 0170 0052, frame 76)
# analyze_test_v1.py is run separately, AFTER the g8 visual note VISUAL_TEST_<run>.txt is written.
set -u
cd /home/kawa/master_project/StereoCrafter
set +u; source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate stereocrafter; set -u
L=scripts/distill/runs/deep_20261004/decoder_ft
T=outputs/deep_20261004/decoder_ft/test_renders
TEST="0042 0052 0125 0128 0141 0147 0170 0204 0225 0251 0259 0301"
STAGE=$1; shift
mkdir -p $T
case $STAGE in
  redec)
    GPU=$1; RUN=$2; SEL=$3; CL=$4; CK=/mnt/ssd_data/deep_20261004/decoder_ft/ck/$RUN
    PAIRS=""; for c in $CL; do PAIRS="$PAIRS,$c:origin_cap,$c:deliv_cap"; done; PAIRS=${PAIRS#,}
    CUDA_VISIBLE_DEVICES=$GPU flock /tmp/claude-gpu$GPU.lock python $L/redecode_v1.py $T stock,$SEL=$CK/step${SEL#s}.pt $PAIRS > $L/redec_test_${RUN}_gpu$GPU.log 2>&1 < /dev/null
    echo "REDECODE [$CL] rc=$? $(date +%F_%T)" ;;
  gates)
    python $L/check_md5_v1.py $L/GATES_C1C2_test.txt $T ${TEST// /,} ;;
  unsharp)
    US=$1; CL=$2
    SIGMA=$(echo $US | sed -E 's/^us_s([0-9.]+)a([0-9.]+)$/\1/'); AMT=$(echo $US | sed -E 's/^us_s([0-9.]+)a([0-9.]+)$/\2/')
    N=0
    for c in $CL; do for lat in origin_cap deliv_cap; do
      python $L/make_unsharp_v1.py $T/${c}_${lat}__stock/${c}_inpainting_results_sbs.mkv $T ${c}_${lat}__${US} $SIGMA $AMT >> $L/unsharp_test.log 2>&1 &
      N=$((N+1)); if [ $((N % 6)) -eq 0 ]; then wait; fi
    done; done; wait
    echo "UNSHARP $US (sigma $SIGMA a $AMT) done $(date +%F_%T)" ;;
  unreg)
    GPU=$1; RUN=$2; SEL=$3; US=$4; CL=$5
    for c in $CL; do
      F=$L/test_unreg_${c}_${RUN}.txt
      [ -e $F ] && { echo "REFUSING: $F exists"; exit 1; }
      ARGS=""; for lat in origin_cap deliv_cap; do for nm in stock $SEL $US; do ARGS="$ARGS $c=$T/${c}_${lat}__${nm}/${c}_inpainting_results_sbs.mkv"; done; done
      SCORE_STEP=4 CUDA_VISIBLE_DEVICES=$GPU flock /tmp/claude-gpu$GPU.lock python scripts/distill/runs/fulldata_v2/beyond4/score_clip_ll.py $ARGS > $F 2>&1 < /dev/null
      echo "UNREG $c rc=$? $(date +%F_%T)"
    done ;;
  rows)
    RUN=$1; SEL=$2; US=$3
    LABS=""; for lat in origin_cap deliv_cap; do for nm in stock $SEL $US; do LABS="$LABS,${lat}__${nm}"; done; done; LABS=${LABS#,}
    PUBL=origin_ll,s25_ll,AYS8_origin_g101,mstudent2_step800_deliv_ll
    python $L/make_rows_v1.py $L/rows_test_${RUN}.json $LABS,$PUBL $L/test_unreg_*_${RUN}.txt --base=scripts/distill/runs/ays_20261004/robust/ROWS_pass1.json:$PUBL ;;
  score)
    GPU=$1; RUN=$2; SEL=$3; CL=$4
    ROWS=$L/rows_test_${RUN}.json
    O=outputs/deep_20261004/decoder_ft/score_reg_test_${RUN}; A=outputs/deep_20261004/decoder_ft/score_aux_test_${RUN}
    TP=outputs/deep_20261004/decoder_ft/temporal_test_${RUN}; PF=outputs/deep_20261004/decoder_ft/pairflicker_test_${RUN}
    mkdir -p $O $A $TP $PF
    for c in $CL; do
      SCORE_STEP=4 CUDA_VISIBLE_DEVICES=$GPU flock /tmp/claude-gpu$GPU.lock python $L/score_registered_df_v1.py $O $ROWS $c >> $L/score_reg_test_${RUN}_gpu$GPU.log 2>&1 < /dev/null
      echo "REG $c rc=$? $(date +%F_%T)"
      CUDA_VISIBLE_DEVICES=$GPU flock /tmp/claude-gpu$GPU.lock env PYTHONPATH=/mnt/ssd_data/deep_20261004/blur_diag/pylib:/mnt/ssd_data/deep_20261004/skeptic/pylib \
        TORCH_HOME=/mnt/ssd_data/deep_20261004/decoder_ft/torch_home HF_HOME=/mnt/ssd_data/deep_20261004/blur_diag/hf_home \
        HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python $L/score_aux_v1.py $A $ROWS $O $c >> $L/score_aux_test_${RUN}_gpu$GPU.log 2>&1 < /dev/null
      echo "AUX $c rc=$? $(date +%F_%T)"
      [ -e $TP/$c.json ] && { echo "REFUSING: $TP/$c.json exists"; exit 1; }
      TA=""; for lab in origin_cap__stock origin_cap__$SEL deliv_cap__stock deliv_cap__$SEL; do TA="$TA $c=$T/${c}_${lab}/${c}_inpainting_results_sbs.mkv"; done
      CUDA_VISIBLE_DEVICES=$GPU flock /tmp/claude-gpu$GPU.lock python scripts/distill/runs/finalcheck_20261004/validate/score_temporal_ll.py $TP/$c.json $TA >> $L/temporal_test_${RUN}_gpu$GPU.log 2>&1 < /dev/null
      echo "TEMPORAL $c rc=$? $(date +%F_%T)"
      PA=""; for lab in origin_cap__stock origin_cap__$SEL deliv_cap__stock deliv_cap__$SEL; do PA="$PA $lab=$T/${c}_${lab}/${c}_inpainting_results_sbs.mkv"; done
      python $L/score_pairflicker_v1.py $PF/$c.json $c $PA >> $L/pairflicker_test_${RUN}.log 2>&1 < /dev/null
      echo "PAIRFLICKER $c rc=$? $(date +%F_%T)"
    done ;;
  roundtrip)
    GPU=$1; RUN=$2; SEL=$3; CK=/mnt/ssd_data/deep_20261004/decoder_ft/ck/$RUN
    CUDA_VISIBLE_DEVICES=$GPU flock /tmp/claude-gpu$GPU.lock env PYTHONPATH=/mnt/ssd_data/deep_20261004/skeptic/pylib TORCH_HOME=/mnt/ssd_data/deep_20261004/decoder_ft/torch_home \
      python $L/roundtrip_v1.py $L/ROUNDTRIP_TEST_${RUN}.json test ${TEST// /,} $SEL=$CK/step${SEL#s}.pt > $L/roundtrip_test_${RUN}.log 2>&1 < /dev/null
    echo "ROUNDTRIP rc=$? $(date +%F_%T)" ;;
  panels)
    RUN=$1; SEL=$2
    python $L/make_panels_v1.py $RUN $SEL > $L/panels_test_${RUN}.log 2>&1
    echo "PANELS rc=$? $(date +%F_%T)" ;;
  *) echo "unknown stage $STAGE"; exit 1 ;;
esac
