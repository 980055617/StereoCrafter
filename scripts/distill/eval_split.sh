#!/bin/bash
# eval_split.sh <mamba_only.pt|ORIGIN> <label>  -> LPIPS on the 12 fulldata_v1 test clips (+0160 continuity), never overwrites outputs
set -u; cd /home/kawa/master_project/StereoCrafter; D=scripts/distill; PY=/home/kawa/miniconda3/envs/stereocrafter/bin/python
CK=$1; LABEL=$2; O=outputs/fulldata/clips; mkdir -p $O $D/runs/fulldata/lpips
TEST=$(python3 -c "import json;print(' '.join(json.load(open('scripts/distill/splits/fulldata_v1.json'))['test']))")
export MAMBA_SELF_ATTN_D_STATE=${MAMBA_SELF_ATTN_D_STATE:-128} MAMBA_SELF_ATTN_EXPAND=1 MAMBA_BIDIRECTIONAL_MODE=fwd MAMBA_SELF_ATTN_REPLACEMENT=gated_residual PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for C in $TEST 0160; do
  IN=video_data/splatting/${C}_splatting_results.mp4
  if [ "$CK" = "ORIGIN" ]; then
    OUTD=$O/${C}_origin; [ -f $OUTD/${C}_inpainting_results_sbs.mp4 ] && continue
    for SRC in outputs/diagnose_0160/clips/${C}_origin outputs/diagnose_0160/origin_base_via_inference_py_guid101; do   # reuse existing origin outputs
      [ -f $SRC/${C}_inpainting_results_sbs.mp4 ] && { mkdir -p $OUTD; ln -sf $(realpath $SRC/${C}_inpainting_results_sbs.mp4) $OUTD/${C}_inpainting_results_sbs.mp4; break; }; done
    [ -f $OUTD/${C}_inpainting_results_sbs.mp4 ] && continue
    mkdir -p $OUTD; MAMBA_SELF_ATTN_INCLUDE='__nomatch__' conda run -n stereocrafter --no-capture-output python3 inpainting_inference.py \
      --config=config/0160_overfit_inference_matched.json --unet_state_path=None --input_video_path="$IN" --save_dir="$OUTD" > "$OUTD.log" 2>&1
  else
    OUTD=$O/${C}_$LABEL; [ -f $OUTD/${C}_inpainting_results_sbs.mp4 ] && continue; mkdir -p $OUTD
    conda run -n stereocrafter --no-capture-output python3 inpainting_inference_hybrid_exclude_up3_attn1.py --unet_state_path="$CK" \
      --include_patterns='down_blocks.0.*,up_blocks.3.*' --exclude_patterns='__nomatch__' --mamba_gate_override=1.0 --input_video_path="$IN" --save_dir="$OUTD" > "$OUTD.log" 2>&1
    grep -q 'updated 5 gated modules' "$OUTD.log" || echo "WARN $C $LABEL: gate override line missing"
  fi
  echo "  infer $C $LABEL done $(date +%H:%M:%S)"
done
[ "$CK" = "ORIGIN" ] && { echo "ORIGIN_READY"; exit 0; }
ARGS=""; for C in $TEST 0160; do ARGS="$ARGS ${C}=$O/${C}_origin/${C}_inpainting_results_sbs.mp4 ${C}=$O/${C}_$LABEL/${C}_inpainting_results_sbs.mp4"; done
conda run -n stereocrafter --no-capture-output python3 $D/score_clip.py $ARGS 2>&1 | grep -viE "warning|setting up|loading model|self.load_state|/home/kawa" | tee $D/runs/fulldata/lpips/$LABEL.txt
python3 - "$LABEL" <<'PY'
import sys,re,json; L=sys.argv[1]; rows={}; cur=None
for l in open(f"scripts/distill/runs/fulldata/lpips/{L}.txt"):
    m=re.match(r'^(\d{4})_(\S+)\s+\(([-\d]+),([-\d]+)\)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)',l)
    if m: rows.setdefault(m.group(1),{})[m.group(2)]={"offset":[int(m.group(3)),int(m.group(4))],"leftPSNR":float(m.group(5)),"lpips":float(m.group(6)),"sharp":float(m.group(7))}
test=json.load(open('scripts/distill/splits/fulldata_v1.json'))['test']
gaps=[rows[c][L]["lpips"]-rows[c]["origin"]["lpips"] for c in test if c in rows and L in rows[c] and "origin" in rows[c]]
summ={"label":L,"n_test":len(gaps),"mean_gap":sum(gaps)/max(len(gaps),1),"worst_gap":max(gaps) if gaps else None,"best_gap":min(gaps) if gaps else None,
      "mean_lpips":sum(rows[c][L]["lpips"] for c in test if c in rows)/max(len(gaps),1),"mean_origin":sum(rows[c]["origin"]["lpips"] for c in test if c in rows)/max(len(gaps),1),
      "gap_0160":(rows.get("0160",{}).get(L,{}).get("lpips",float('nan'))-rows.get("0160",{}).get("origin",{}).get("lpips",float('nan'))),"rows":rows}
json.dump(summ,open(f"scripts/distill/runs/fulldata/lpips/{L}.json","w"),indent=1)
print(f"LPIPS_SUMMARY {L}: n={summ['n_test']} mean={summ['mean_lpips']:.4f} origin={summ['mean_origin']:.4f} mean_gap={summ['mean_gap']:+.4f} worst={summ['worst_gap']:+.4f} best={summ['best_gap']:+.4f} gap0160={summ['gap_0160']:+.4f}")
PY
echo "EVAL_END $LABEL $(date +%H:%M:%S)"
