#!/bin/bash
# Teacher-forced capture of the fulldata_v1 windows on the CORRECTED splatting, 2 GPUs in parallel (chunk parity), cache fulldata_tf_v2.
set -u; cd /home/kawa/master_project/StereoCrafter; D=scripts/distill; M=$D/runs/fulldata/manifests_v2; R=$D/runs/fulldata_v2; mkdir -p $R/cap
set +u; source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate stereocrafter; set -u
BEST=/mnt/ssd_data/stereocrafter_weights/_distill_injected/light_lvl0_fulldata333_8k_mamba_only.pt; OUT=/mnt/ssd_data/attn_cache/fulldata_tf_v2
N=$(ls $M/cap_*.json | wc -l)
lane() { G=$1; for k in $(seq $G 2 $((N-1))); do K=$(printf %03d $k); [ -f $R/cap/cap_$K.done ] && continue
  df -BG /mnt/ssd_data | awk 'NR==2{gsub("G","",$4); if(($4+0)<150){exit 1}}' || { echo "DISK_LOW gpu$G"; return 1; }
  CUDA_VISIBLE_DEVICES=$G CAP_MANIFEST=$M/cap_$K.json CAP_OUT=$OUT CKPT=$BEST CAP_TEMB_REF=$D/runs/fulldata/temb_ref_origin.pt python $D/capture_fulldata.py > $R/cap/cap_$K.log 2>&1
  L=$(grep -E '^\[capture\] DONE' $R/cap/cap_$K.log | tail -1); if [ -z "$L" ]; then echo "CHUNK_FAIL $K gpu$G"; grep -E 'Traceback|Error|ABORT|REFUSED|assert' $R/cap/cap_$K.log | tail -2; return 1; fi
  echo "gpu$G chunk $K $L"; touch $R/cap/cap_$K.done; done; }
echo "CAPTURE_V2_START chunks=$N $(date +%H:%M:%S)"; lane 0 & lane 1 & wait
echo "CAPTURE_V2_DONE $(date +%H:%M:%S) $(ls $R/cap/*.done | wc -l)/$N chunks, $(du -sh $OUT | cut -f1)"
