#!/bin/bash
cd /home/kawa/master_project/StereoCrafter; D=scripts/distill; O=$D/runs/fulldata_v2/onpolicy
set +u; source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate stereocrafter; set -u
CK=/mnt/ssd_data/stereocrafter_weights/_distill_injected/light_lvl0_fulldata333_8k_mamba_only.pt
run() { C=$1; G=$2; CUDA_VISIBLE_DEVICES=$G CAP_ONPOLICY=1 CAP_SAVE=0 CAP_MAXCHUNKS=6 CAP_OUT=$O/$C CKPT=$CK CAP_VIDEO=video_data/splatting/${C}_splatting_results.mp4 python $D/capture_attn.py > $O/$C.log 2>&1; echo "$C $(grep -E 'ON-POLICY' $O/$C.log | cut -c1-300)"; }
( run 0160 0; run 0042 0 ) & ( run 0204 1; run 0170 1 ) & wait
echo ONPOL_V2_DONE
