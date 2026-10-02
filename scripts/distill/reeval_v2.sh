#!/bin/bash
# Re-evaluate origin and the 333-clip Mamba on the 12 test clips + 0160 against the REAL right-eye GT (Plan Y data).
# iPhone clips (0160-0309) need fresh inference (their splatting input changed); AVP clips reuse existing outputs.
set -u; cd /home/kawa/master_project/StereoCrafter; D=scripts/distill; O=outputs/fulldata_v2/clips; R=$D/runs/fulldata_v2
set +u; source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate stereocrafter; set -u
export MAMBA_SELF_ATTN_D_STATE=128 MAMBA_SELF_ATTN_EXPAND=1 MAMBA_BIDIRECTIONAL_MODE=fwd MAMBA_SELF_ATTN_REPLACEMENT=gated_residual PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CK=/mnt/ssd_data/stereocrafter_weights/_distill_injected/light_lvl0_fulldata333_8k_mamba_only.pt
TEST="0042 0052 0125 0128 0141 0147 0170 0204 0225 0251 0259 0301"
infer() { # infer <clip> <label> <gpu>
  local C=$1 L=$2 G=$3 OD=$O/${C}_$2; mkdir -p $OD; [ -f $OD/${C}_inpainting_results_sbs.mp4 ] && return 0
  if [ "$((10#$C))" -lt 160 ] && [ -f outputs/fulldata/clips/${C}_$L/${C}_inpainting_results_sbs.mp4 ]; then ln -sfn $(realpath outputs/fulldata/clips/${C}_$L/${C}_inpainting_results_sbs.mp4) $OD/${C}_inpainting_results_sbs.mp4; echo "  $C $L reused (AVP input unchanged)"; return 0; fi
  if [ "$L" = "origin" ]; then CUDA_VISIBLE_DEVICES=$G MAMBA_SELF_ATTN_INCLUDE='__nomatch__' python inpainting_inference.py --config=config/0160_overfit_inference_matched.json --unet_state_path=None --input_video_path=video_data/splatting/${C}_splatting_results.mp4 --save_dir=$OD > $OD.log 2>&1
  else CUDA_VISIBLE_DEVICES=$G python inpainting_inference_hybrid_exclude_up3_attn1.py --unet_state_path=$CK --include_patterns='down_blocks.0.*,up_blocks.3.*' --exclude_patterns='__nomatch__' --mamba_gate_override=1.0 --input_video_path=video_data/splatting/${C}_splatting_results.mp4 --save_dir=$OD > $OD.log 2>&1; fi
  echo "  $C $L inferred on gpu$G $(date +%H:%M:%S)"; }
( for C in $TEST 0160; do infer $C origin 0; done ) & ( for C in $TEST 0160; do infer $C all_8k 1; done ) & wait
ARGS=""; for C in $TEST 0160; do ARGS="$ARGS ${C}=$O/${C}_origin/${C}_inpainting_results_sbs.mp4 ${C}=$O/${C}_all_8k/${C}_inpainting_results_sbs.mp4"; done
echo "=== REAL-GT LPIPS (train = regenerated bundles) ==="; python $D/score_clip.py $ARGS 2>&1 | grep -viE "warning|setting up|loading model|self.load_state|/home/kawa" | tee $R/lpips_realgt.txt
echo "REEVAL_V2_DONE $(date +%H:%M:%S)"
