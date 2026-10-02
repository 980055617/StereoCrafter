#!/bin/bash
# After everything else: the Mamba-side oracle gate on 2 clips (one per regime), lossless.
set -u; cd /home/kawa/master_project/StereoCrafter
PY=/home/kawa/miniconda3/envs/stereocrafter/bin/python
until ! systemctl --user is-active --quiet sk-ext0 && ! systemctl --user is-active --quiet sk-ext1 \
      && ! systemctl --user is-active --quiet sk-gpu0 && ! systemctl --user is-active --quiet sk-gpu1 \
      && ! systemctl --user is-active --quiet sk-ctrl; do sleep 20; done
MAMBA=/mnt/ssd_data/stereocrafter_weights/_distill_injected/light_lvl0_fulldata333_v2_8k_mamba_only.pt
LOG=outputs/skeptic1_stack/timing_oraclem.txt
export CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LOSSLESS_SBS=1 KEEP_ANAGLYPH=0
export MAMBA_SELF_ATTN_INCLUDE='down_blocks.0.*,up_blocks.3.*' MAMBA_SELF_ATTN_EXCLUDE='__nomatch__'
export MAMBA_SELF_ATTN_D_STATE=128 MAMBA_SELF_ATTN_EXPAND=1
export MAMBA_BIDIRECTIONAL_MODE=fwd MAMBA_SELF_ATTN_REPLACEMENT=gated_residual
export SK_UNET=$MAMBA BD_M=4 BD_SUBST=4,5,6
for CL in 0301 0052; do
  OD=outputs/skeptic1_stack/clips/${CL}_mamba_oracle456_ll
  compgen -G "$OD/*_sbs.mkv" >/dev/null && { echo "SKIP $CL" | tee -a $LOG; continue; }
  mkdir -p $OD; T0=$(date +%s)
  SK_CLIP=$CL SK_OUT=$OD $PY scripts/distill/runs/skeptic1/oracle_mamba_v1.py > $OD.log 2>&1
  echo "RUN $CL mamba_oracle456_ll rc=$? secs=$(( $(date +%s) - T0 )) $(date +%H:%M:%S) dir=$OD" | tee -a $LOG
done
echo ORACLEM_DONE
