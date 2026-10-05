#!/bin/bash
# blur_diag (deep_20261004) main chain, GPU 0 only; EVERY GPU process under flock /tmp/claude-gpu0.lock (one acquisition
# per step so other lanes interleave).  Launched only after the 0204 scoring smoke passed G0.  The VAE round trips run in
# chain_vae_r2.sh; score() waits until a clip's VAE rows are complete.  Order: G3 identity control -> HIRES_B smoke ->
# HIRES_B full renders (0170, 0042) -> scoring of the 11 non-smoke clips.
# Every output dir is new; scorers and writers refuse to overwrite.  Detail statistics run on CPU without the lock.
set -u
set +u; source ~/miniconda3/etc/profile.d/conda.sh; conda activate stereocrafter; set -u
cd /home/kawa/master_project/StereoCrafter
R=scripts/distill/runs/deep_20261004/blur_diag
O=outputs/deep_20261004/blur_diag
S=$O/scores_r2
L=/mnt/ssd_data/deep_20261004/blur_diag
LK=/tmp/claude-gpu0.lock
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NRENV="PYTHONPATH=$L/pylib TORCH_HOME=$L/torch_home HF_HOME=$L/hf_home HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1"
mkdir -p $S/lpips $S/detail $S/nr $S/logs $O/hiresB_r2/clips
echo "CHAIN_MAIN_START $(date +%F_%T)"

vae() {   # clip
  if [ -e $O/vae_rt_r2/$1/meta_vae_rt.json ] && grep -q total_seconds $O/vae_rt_r2/$1/meta_vae_rt.json; then
    echo "VAE $1 already complete -> skip"; return; fi
  flock $LK python $R/vae_roundtrip_r2.py $1 > $O/vae_rt_r2/$1.log 2>&1
  echo "VAE $1 rc=$? $(date +%F_%T) $(grep -c '] wrote ' $O/vae_rt_r2/$1.log) rows written"
}
score() { # clip  (LPIPS + NR in one lock acquisition; detail on CPU in the background)
  until [ -e $O/vae_rt_r2/$1/meta_vae_rt.json ] && grep -q total_seconds $O/vae_rt_r2/$1/meta_vae_rt.json; do sleep 30; done
  ( CUDA_VISIBLE_DEVICES='' python $R/score_detail_r2.py $S/detail $1 > $S/logs/detail_$1.log 2>&1;
    echo "DETAIL $1 rc=$? $(date +%F_%T)" ) &
  flock $LK bash -c "python $R/score_lpips_r2.py $S/lpips $1 > $S/logs/lpips_$1.log 2>&1; echo LPIPS_RC=\$? >> $S/logs/lpips_$1.log; \
    env $NRENV python $R/score_nr_r2.py $S/nr $1 > $S/logs/nr_$1.log 2>&1; echo NR_RC=\$? >> $S/logs/nr_$1.log"
  echo "SCORE $1 $(grep -o 'G0 [A-Z]*' $S/logs/lpips_$1.log | tail -1) $(grep -o 'LPIPS_RC=[0-9]*' $S/logs/lpips_$1.log) $(grep -o 'NR_RC=[0-9]*' $S/logs/nr_$1.log) $(date +%F_%T)"
}
render_b() { # clip H W maxchunks label
  local OD=$O/hiresB_r2/clips/$1_$5
  if [ -e $OD ] || [ -e $OD.log ]; then echo "RENDER $1 $5 exists -> skip"; return; fi
  mkdir -p $OD
  local T0=$(date +%s)
  ( unset MAMBA_SELF_ATTN_EXCLUDE MAMBA_SELF_ATTN_D_STATE MAMBA_SELF_ATTN_EXPAND MAMBA_BIDIRECTIONAL_MODE MAMBA_SELF_ATTN_REPLACEMENT
    export MAMBA_SELF_ATTN_INCLUDE='__nomatch__' LOSSLESS_SBS=1 KEEP_ANAGLYPH=0 UP_CLIP=$1 UP_OUT=$OD UP_H=$2 UP_W=$3 UP_MAXCHUNKS=$4
    flock $LK python $R/infer_upwin_r2.py > $OD.log 2>&1 )
  local RC=$?
  local G=PASS
  grep -q "^\[lossless\] wrote FFV1 " $OD.log || G=FAIL_nowrite
  grep -q "res=$2x$3" $OD.log || G=FAIL_res
  grep -q "unet=None" $OD.log || G=FAIL_unet
  grep -q "Partial UNet state load" $OD.log && G=FAIL_partialload
  echo "RENDER $1 $5 rc=$RC gate=$G secs=$(( $(date +%s) - T0 )) $(grep -oE 'md5\(pre-encode\)=[0-9a-f]+' $OD.log | tail -1) $(date +%F_%T) dir=$OD"
}

# ---- G3 identity control (576x1024, 1 window) on 0170 vs the existing GPU-0 origin_ll render
render_b 0170 576 1024 1 ident_576x1024_1chunk
python $R/check_identity_r2.py $O/hiresB_r2/clips/0170_ident_576x1024_1chunk/0170_inpainting_results_sbs.mkv \
  outputs/beyond4_lossless/clips/0170_origin_ll/0170_inpainting_results_sbs.mkv > $O/hiresB_r2/G3_identity_0170.txt 2>&1
G3=$(grep -o 'G3_IDENTITY [A-Z]*' $O/hiresB_r2/G3_identity_0170.txt)
echo "$G3 $(date +%F_%T)"
# HIRES_B proceeds either way; a non-bit-exact G3 is reported as a caveat (PREREG G3)
render_b 0170 1024 1792 1 smoke_upx175_1chunk
render_b 0170 1024 1792 "" origin_upx175
render_b 0042 1024 1792 "" origin_upx175
# PREREG ADDENDUM 2 (R4-L): LANCZOS-downsampled copies of the two HIRES_B renders (CPU only, before their scoring)
CUDA_VISIBLE_DEVICES='' python $R/make_lanczos_rows_r2.py --hb 0170 0042 > $R/make_lanczos_hb_r2.log 2>&1
echo "LANCZOS_HB rc=$? $(grep -c '^wrote' $R/make_lanczos_hb_r2.log) files $(date +%F_%T)"
for c in 0052 0125 0128 0141 0147 0225 0251 0259 0301 0170 0042; do score $c; done
wait
echo "CHAIN_MAIN_DONE $(date +%F_%T)"
