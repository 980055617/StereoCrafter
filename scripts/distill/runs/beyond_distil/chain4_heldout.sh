#!/bin/bash
# CHAIN 4 (GPU 1): HELD-OUT generalization.  Sample the DEPLOYED config (8 steps, guidance 1.01) with the
# smoke1 student swapped in, on clips it was NEVER trained on -- two where origin is ALREADY SHARPER than GT
# (0052, 0128: the over-sharpen risk) and two where origin is under-sharp (0125, 0170).  Scored against the
# deployed origin and s25 reference rows that already exist in outputs/fulldata_v2/clips/.
#   BD_HCK  checkpoint to sample   BD_HNAME  tag   BD_HCLIPS  clips
set -u; cd /home/kawa/master_project/StereoCrafter
set +u; source ~/miniconda3/etc/profile.d/conda.sh; conda activate stereocrafter; set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=1
H=scripts/distill/runs/beyond_distil; MF=scripts/distill/runs/diag_trainer/minift
newdir(){ local O=$1; local k=1; while [ -e "$O" ]; do k=$((k+1)); O=$1_$k; done; echo $O; }
CK=${BD_HCK:-scripts/distill/runs/beyond_distil/smoke1/step800.pt}
NAME=${BD_HNAME:-smoke1s800}
HCLIPS=${BD_HCLIPS:-"0052 0128 0125 0170"}
echo "checkpoint $CK  tag $NAME  held-out clips $HCLIPS"
ARGS=""
for C in $HCLIPS; do
  ARGS="$ARGS $C=outputs/fulldata_v2/clips/${C}_origin/${C}_inpainting_results_sbs.mp4"
  ARGS="$ARGS $C=outputs/fulldata_v2/clips/${C}_origin_s25/${C}_inpainting_results_sbs.mp4"
  O=$(newdir outputs/beyond_distil/${C}_heldout_${NAME})
  echo "START infer $C -> $O $(date +%T)"
  MINIFT_CK=$CK python $MF/xcheck_hybrid_minift.py e1all $C $O > ${O}.log 2>&1
  echo "EXIT infer $C $? $(date +%T)"
  F=$O/${C}_inpainting_results_sbs.mp4
  [ -f "$F" ] && ARGS="$ARGS $C=$F" || echo "MISSING $F"
done
echo "START score $(date +%T)"
python scripts/distill/score_clip.py $ARGS 2>&1 \
  | grep -viE "warning|setting up|loading model|self.load_state|/home/kawa" | tee $H/scores_heldout_${NAME}.txt
echo "EXIT score $? $(date +%T)"; echo CHAIN4_DONE
