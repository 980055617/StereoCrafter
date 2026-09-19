#!/bin/bash
# GPU0: teacher-forced capture of all fulldata_v1 windows in curve-prefix order (+ .ready markers), high-res capture, 'all' fits, arms.
set -u; cd /home/kawa/master_project/StereoCrafter; D=scripts/distill; PY=/home/kawa/miniconda3/envs/stereocrafter/bin/python
BEST=/mnt/ssd_data/stereocrafter_weights/_distill_injected/light_lvl0_multiclip13_r2_mamba_only.pt
M=$D/runs/fulldata/manifests; R=$D/runs/fulldata; F=$R/fits; export CUDA_VISIBLE_DEVICES=0
N=$(python3 -c "import json;print(json.load(open('$M/prefixes.json'))['n_chunks'])")
declare -A LAST; while IFS== read k v; do LAST[$k]=$v; done < <(python3 -c "import json;[print(f'{k}={v}') for k,v in json.load(open('$M/prefixes.json'))['prefix_last_chunk'].items()]")
echo "CAPTURE_LANE_START chunks=$N $(date +%H:%M:%S)"
for k in $(seq 0 $((N-1))); do
  K=$(printf %03d $k); [ -f $R/cap_$K.done ] && continue
  df -BG /mnt/ssd_data | awk 'NR==2{gsub("G","",$4); if(($4+0)<150){print "DISK_LOW "$4"G"; exit 1}}' || { echo "CAPTURE_ABORT disk"; exit 1; }
  G=$(python3 -c "import json;print(json.load(open('$M/cap_$K.json'))['entries'][0]['group'])")
  OUT=/mnt/ssd_data/attn_cache/fulldata_tf; [ "$G" = "c13w14" ] && OUT=/mnt/ssd_data/attn_cache/fulldata_tf_c13w14
  CAP_MANIFEST=$M/cap_$K.json CAP_OUT=$OUT CKPT=$BEST $PY $D/capture_fulldata.py > $R/cap_$K.log 2>&1
  rc=$?; L=$(grep -E '^\[capture\] DONE' $R/cap_$K.log | tail -1)
  if [ $rc -ne 0 ] || [ -z "$L" ]; then echo "CHUNK_FAIL $K rc=$rc"; grep -E 'Traceback|Error|ABORT|REFUSED|assert' $R/cap_$K.log | tail -3; exit 1; fi
  echo "chunk $K ($G) $L"; touch $R/cap_$K.done
  for p in "${!LAST[@]}"; do [ "${LAST[$p]}" = "$k" ] && { touch $R/prefix_$p.ready; echo "PREFIX_READY $p $(date +%H:%M:%S)"; }; done
done
echo "CAPTURE_DONE $(date +%H:%M:%S)  $(du -sh /mnt/ssd_data/attn_cache/fulldata_tf | cut -f1)"
for H in $M/hires_*.json; do
  K=$(basename $H .json); [ -f $R/$K.done ] && continue
  CAP_MANIFEST=$H CAP_OUT=/mnt/ssd_data/attn_cache/fulldata_tf_hires CKPT=$BEST CAP_RES=1024x1792 $PY $D/capture_fulldata.py > $R/$K.log 2>&1
  echo "hires $K $(grep -E '^\[capture\] DONE|ABORT|Traceback' $R/$K.log | tail -1)"; touch $R/$K.done
done
echo "HIRES_CAPTURE_DONE $(date +%H:%M:%S)"
$D/fulldata_fit.sh all all_8k
$D/fulldata_fit.sh all all_24k STEPS=24000 EVAL_EVERY=2000
$D/fulldata_fit.sh all all_8k_lr1e-3 LR=1e-3
$D/fulldata_fit.sh all all_8k_hires HIRES_CACHE=/mnt/ssd_data/attn_cache/fulldata_tf_hires HIRES_P=0.25
$D/fulldata_fit.sh all all_8k_ds32 D_STATE=32
echo "CAPTURE_LANE_DONE $(date +%H:%M:%S)"
