#!/bin/bash
# eval_v2.sh <mamba_only.pt> <label> : fresh inference of the 12 test clips + 0160 on the CORRECTED splatting (2 GPUs), then real-GT LPIPS vs origin.
set -u; cd /home/kawa/master_project/StereoCrafter; D=scripts/distill; O=outputs/fulldata_v2/clips; R=$D/runs/fulldata_v2
set +u; source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate stereocrafter; set -u
export MAMBA_SELF_ATTN_D_STATE=${MAMBA_SELF_ATTN_D_STATE:-128} MAMBA_SELF_ATTN_EXPAND=1 MAMBA_BIDIRECTIONAL_MODE=fwd MAMBA_SELF_ATTN_REPLACEMENT=gated_residual PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CK=$1; L=$2; TEST="0042 0052 0125 0128 0141 0147 0170 0204 0225 0251 0259 0301"
infer() { C=$1 G=$2; OD=$O/${C}_$L; mkdir -p $OD; [ -f $OD/${C}_inpainting_results_sbs.mp4 ] && return 0
  CUDA_VISIBLE_DEVICES=$G python inpainting_inference_hybrid_exclude_up3_attn1.py --unet_state_path=$CK --include_patterns='down_blocks.0.*,up_blocks.3.*' --exclude_patterns='__nomatch__' --mamba_gate_override=1.0 --input_video_path=video_data/splatting/${C}_splatting_results.mp4 --save_dir=$OD > $OD.log 2>&1
  grep -q 'updated 5 gated modules' $OD.log || echo "WARN $C: gate line missing"; echo "  $C $L gpu$G $(date +%H:%M:%S)"; }
( for C in 0042 0052 0125 0128 0141 0147 0160; do infer $C 0; done ) & ( for C in 0170 0204 0225 0251 0259 0301; do infer $C 1; done ) & wait
ARGS=""; for C in $TEST 0160; do ARGS="$ARGS ${C}=$O/${C}_origin/${C}_inpainting_results_sbs.mp4 ${C}=$O/${C}_$L/${C}_inpainting_results_sbs.mp4"; done
python $D/score_clip.py $ARGS 2>&1 | grep -viE "warning|setting up|loading model|self.load_state|/home/kawa" | tee $R/lpips_$L.txt
python - "$L" <<'PY'
import re,sys; L=sys.argv[1]; rows={}
for l in open(f"scripts/distill/runs/fulldata_v2/lpips_{L}.txt"):
    m=re.match(r'^(\d{4})_(\S+)\s+\S+\s+[\d.]+\s+([\d.]+)',l)
    if m: rows.setdefault(m.group(1),{})[m.group(2)]=float(m.group(3))
T=[c for c in rows if c!='0160' and L in rows[c] and 'origin' in rows[c]]; d=[rows[c][L]-rows[c]['origin'] for c in T]
print(f"REALGT_SUMMARY {L}: n={len(T)} origin={sum(rows[c]['origin'] for c in T)/len(T):.4f} model={sum(rows[c][L] for c in T)/len(T):.4f} gap={sum(d)/len(d):+.4f} worst={max(d):+.4f} best={min(d):+.4f} 0160={rows.get('0160',{}).get(L,float('nan'))-rows.get('0160',{}).get('origin',float('nan')):+.4f}")
PY
echo "EVAL_V2_DONE $L $(date +%H:%M:%S)"
