#!/bin/bash
# PHASE 0 for the Mamba-side run: the construction controls, then the FULL oracle (BD_SUBST=all) on 0301,
# then the 4,5,6 oracle on the two clips skeptic1 did not cover (0204, 0147).
# Every run gets its own NEW directory; nothing is ever overwritten.
set -u
cd /home/kawa/master_project/StereoCrafter
PY=/home/kawa/miniconda3/envs/stereocrafter/bin/python
D=scripts/distill/runs/beyond_distil_mamba
MAMBA=/mnt/ssd_data/stereocrafter_weights/_distill_injected/light_lvl0_fulldata333_v2_8k_mamba_only.pt
OUT=outputs/beyond_distil_mamba
LOG=$D/chain0.log
mkdir -p $OUT/clips
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LOSSLESS_SBS=1 KEEP_ANAGLYPH=0

echo "=== [1/3] construction controls $(date +%T) ===" | tee -a $LOG
VM_CLIP=0301 VM_M=4 VM_WINS=0 VM_SLOTS=up3 $PY $D/verify_mamba_v1.py > $D/verify_0301.log 2>&1
echo "verify rc=$? $(date +%T)" | tee -a $LOG
tail -30 $D/verify_0301.log | tee -a $LOG
grep -q VM_VERIFY_DONE $D/verify_0301.log || { echo "CONTROL FAILED - stopping chain0" | tee -a $LOG; exit 3; }

# the oracle rows use the shipped Mamba sampling path (skeptic1/oracle_mamba_v1.py), unmodified
export MAMBA_SELF_ATTN_INCLUDE='down_blocks.0.*,up_blocks.3.*' MAMBA_SELF_ATTN_EXCLUDE='__nomatch__'
export MAMBA_SELF_ATTN_D_STATE=128 MAMBA_SELF_ATTN_EXPAND=1
export MAMBA_BIDIRECTIONAL_MODE=fwd MAMBA_SELF_ATTN_REPLACEMENT=gated_residual
export SK_UNET=$MAMBA BD_M=4

echo "=== [2/3] FULL oracle (all 8 steps) on 0301 $(date +%T) ===" | tee -a $LOG
OD=$OUT/clips/0301_mamba_oracleALL_ll
if ! compgen -G "$OD/*_sbs.mkv" >/dev/null; then
  mkdir -p $OD; T0=$(date +%s)
  BD_SUBST=all SK_CLIP=0301 SK_OUT=$OD $PY scripts/distill/runs/skeptic1/oracle_mamba_v1.py > $OD.log 2>&1
  echo "RUN 0301 mamba_oracleALL_ll rc=$? secs=$(( $(date +%s) - T0 )) dir=$OD" | tee -a $LOG
else echo "SKIP 0301 oracleALL exists" | tee -a $LOG; fi

echo "=== [3/3] oracle456 on 0204 and 0147 $(date +%T) ===" | tee -a $LOG
for CL in 0204 0147; do
  OD=$OUT/clips/${CL}_mamba_oracle456_ll
  if compgen -G "$OD/*_sbs.mkv" >/dev/null; then echo "SKIP $CL" | tee -a $LOG; continue; fi
  mkdir -p $OD; T0=$(date +%s)
  BD_SUBST=4,5,6 SK_CLIP=$CL SK_OUT=$OD $PY scripts/distill/runs/skeptic1/oracle_mamba_v1.py > $OD.log 2>&1
  echo "RUN $CL mamba_oracle456_ll rc=$? secs=$(( $(date +%s) - T0 )) dir=$OD" | tee -a $LOG
done
echo "CHAIN0_DONE $(date +%T)" | tee -a $LOG
