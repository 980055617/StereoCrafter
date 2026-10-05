#!/bin/bash
# blur_diag render batch (PREREG ADDENDUM 3): ONE acquisition of /tmp/claude-gpu0.lock for
#   G3 identity control (0170, 576x1024, 1 window) -> HIRES_B smoke (0170, 1024x1792, 1 window) -> HIRES_B full 0170, 0042
# then, outside the lock, the LANCZOS copies (PREREG ADDENDUM 2, CPU).  ORIGIN only: unet_state_path None,
# MAMBA_SELF_ATTN_INCLUDE='__nomatch__' (as every origin_ll render).  Every run dir is new.
set -u
set +u; source ~/miniconda3/etc/profile.d/conda.sh; conda activate stereocrafter; set -u
cd /home/kawa/master_project/StereoCrafter
R=scripts/distill/runs/deep_20261004/blur_diag
O=outputs/deep_20261004/blur_diag
mkdir -p $O/hiresB_r2/clips
echo "RENDER_BATCH_START $(date +%F_%T)"
flock /tmp/claude-gpu0.lock bash -c '
set -u
R=scripts/distill/runs/deep_20261004/blur_diag; O=outputs/deep_20261004/blur_diag; T=$O/hiresB_r2/timing_gpu0.txt
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True LOSSLESS_SBS=1 KEEP_ANAGLYPH=0
unset MAMBA_SELF_ATTN_EXCLUDE MAMBA_SELF_ATTN_D_STATE MAMBA_SELF_ATTN_EXPAND MAMBA_BIDIRECTIONAL_MODE MAMBA_SELF_ATTN_REPLACEMENT
export MAMBA_SELF_ATTN_INCLUDE=__nomatch__
echo "RENDER_BATCH_LOCKED $(date +%F_%T)" | tee -a $T
rb() { # clip H W maxchunks label
  local OD=$O/hiresB_r2/clips/$1_$5
  if [ -e $OD ] || [ -e $OD.log ]; then echo "RENDER $1 $5 exists -> skip" | tee -a $T; return; fi
  mkdir -p $OD
  local T0=$(date +%s)
  UP_CLIP=$1 UP_OUT=$OD UP_H=$2 UP_W=$3 UP_MAXCHUNKS=$4 python $R/infer_upwin_r2.py > $OD.log 2>&1
  local RC=$? G=PASS
  grep -q "^\[lossless\] wrote FFV1 " $OD.log || G=FAIL_nowrite
  grep -q "res=$2x$3" $OD.log || G=FAIL_res
  grep -q "unet=None" $OD.log || G=FAIL_unet
  grep -q "Partial UNet state load" $OD.log && G=FAIL_partialload
  echo "RENDER $1 $5 rc=$RC gate=$G secs=$(( $(date +%s) - T0 )) $(grep -oE "md5\(pre-encode\)=[0-9a-f]+" $OD.log | tail -1) $(date +%F_%T) dir=$OD" | tee -a $T
}
rb 0170 576 1024 1 ident_576x1024_1chunk_b
python $R/check_identity_r2.py $O/hiresB_r2/clips/0170_ident_576x1024_1chunk_b/0170_inpainting_results_sbs.mkv \
  outputs/beyond4_lossless/clips/0170_origin_ll/0170_inpainting_results_sbs.mkv > $O/hiresB_r2/G3_identity_0170.txt 2>&1
echo "$(grep -o "G3_IDENTITY [A-Z]*" $O/hiresB_r2/G3_identity_0170.txt) $(date +%F_%T)" | tee -a $T
rb 0170 1024 1792 1 smoke_upx175_1chunk
rb 0170 1024 1792 "" origin_upx175
rb 0042 1024 1792 "" origin_upx175
echo "RENDER_BATCH_UNLOCK $(date +%F_%T)" | tee -a $T'
CUDA_VISIBLE_DEVICES='' python $R/make_lanczos_rows_r2.py --hb 0170 0042 > $R/make_lanczos_hb_r2.log 2>&1
echo "LANCZOS_HB rc=$? $(grep -c '^wrote' $R/make_lanczos_hb_r2.log) files $(date +%F_%T)"
echo "RENDER_BATCH_DONE $(date +%F_%T)"
