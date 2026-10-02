#!/bin/bash
# after the control fine-tune (origin's own 5 attn1 slots): origin-mode inference with the fine-tuned full state, real-GT LPIPS vs origin
set -u; cd /home/kawa/master_project/StereoCrafter; D=scripts/distill; R=$D/runs/fulldata_v2; O=outputs/fulldata_v2/clips
set +u; source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate stereocrafter; set -u
until ! systemctl --user is-active --quiet 'originattn_ctrl2_*'; do sleep 300; done
W=$(ls -d /mnt/ssd_data/stereocrafter_weights/GTfinetune_v2_originattn_control/MambaCrafter_*/ | tail -1); echo "run dir $W"; RK=$(ls -t logs/*_rank0.log | head -1); grep -oE 'Epoch [0-9]+/2 done \| avg_loss=[0-9.]+' "$RK" | tr '\n' ' '; echo
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; TEST="0042 0052 0125 0128 0141 0147 0170 0204 0225 0251 0259 0301"
for E in 002 001; do CK=${W}train_state_epoch000$E.pt; [ -f $CK ] || continue; L=originattn_e$E
  infer() { C=$1 G=$2; OD=$O/${C}_$L; mkdir -p $OD; [ -f $OD/${C}_inpainting_results_sbs.mp4 ] && return 0
    CUDA_VISIBLE_DEVICES=$G MAMBA_SELF_ATTN_INCLUDE='__nomatch__' python inpainting_inference.py --config=config/0160_overfit_inference_matched.json --unet_state_path=$CK --expected_partial_unet_state=True --input_video_path=video_data/splatting/${C}_splatting_results.mp4 --save_dir=$OD > $OD.log 2>&1; }
  ( for C in 0042 0052 0125 0128 0141 0147 0160; do infer $C 0; done ) & ( for C in 0170 0204 0225 0251 0259 0301; do infer $C 1; done ) & wait
  ARGS=""; for C in $TEST 0160; do ARGS="$ARGS ${C}=$O/${C}_origin/${C}_inpainting_results_sbs.mp4 ${C}=$O/${C}_$L/${C}_inpainting_results_sbs.mp4"; done
  python $D/score_clip.py $ARGS 2>&1 | grep -viE "warning|setting up|loading model|self.load_state|/home/kawa" > $R/lpips_$L.txt
  python - "$L" <<'PY'
import re,sys; L=sys.argv[1]; rows={}
for l in open(f"scripts/distill/runs/fulldata_v2/lpips_{L}.txt"):
    m=re.match(r'^(\d{4})_(\S+)\s+\S+\s+[\d.]+\s+([\d.]+)',l)
    if m: rows.setdefault(m.group(1),{})[m.group(2)]=float(m.group(3))
T=[c for c in rows if c!='0160' and L in rows[c]]; d=[rows[c][L]-rows[c]['origin'] for c in T]
print(f"REALGT_SUMMARY {L}: n={len(T)} origin={sum(rows[c]['origin'] for c in T)/len(T):.4f} model={sum(rows[c][L] for c in T)/len(T):.4f} gap={sum(d)/len(d):+.4f} worst={max(d):+.4f} best={min(d):+.4f} 0160={rows.get('0160',{}).get(L,float('nan'))-rows.get('0160',{}).get('origin',float('nan')):+.4f}")
PY
done
rm -rf ${W}deepspeed_state_* ${W}train_state_latest.pt 2>/dev/null; echo "CTRL_EVAL_DONE $(date +%H:%M:%S)"
