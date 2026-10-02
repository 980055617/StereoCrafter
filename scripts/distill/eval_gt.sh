#!/bin/bash
# eval_gt.sh <train_state.pt|mamba_only.pt> <label> : 12 test clips + 0160 on GPU1 -> whole-frame LPIPS (score_clip) and
# composite / inside-mask metrics (score_composite) for origin, the distilled student (all_8k) and <label>.
set -u; cd /home/kawa/master_project/StereoCrafter; D=scripts/distill; R=$D/runs/fulldata/beyond; export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
CK=$1; L=$2; $D/eval_split.sh "$CK" "$L" 2>&1 | grep -E 'LPIPS_SUMMARY|WARN|Traceback'
TEST=$(python3 -c "import json;print(' '.join(json.load(open('scripts/distill/splits/fulldata_v1.json'))['test']))"); ARGS=""
for C in $TEST 0160; do for K in origin all_8k $L; do ARGS="$ARGS ${C}=outputs/fulldata/clips/${C}_$K/${C}_inpainting_results_sbs.mp4"; done; done
conda run -n stereocrafter --no-capture-output python3 $D/score_composite.py $ARGS 2>&1 | grep -vE 'Warning|warn|Setting up|Loading model|/home/kawa|load_state_dict' | tee $R/composite_$L.txt
python3 - "$L" <<'PY'
import re,sys,collections; L=sys.argv[1]; rows=collections.defaultdict(dict)
for l in open(f"scripts/distill/runs/fulldata/beyond/composite_{L}.txt"):
    m=re.match(r'^(\d{4})_(\S+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)',l)
    if m: rows[m.group(1)][m.group(2)]=tuple(map(float,m.groups()[2:]))
T=[c for c in rows if c!='0160']
for k in ("origin","all_8k",L):
    if all(k in rows[c] for c in T): print(f"GT_SUMMARY {k:14s} raw={sum(rows[c][k][0] for c in T)/len(T):.4f} composite={sum(rows[c][k][1] for c in T)/len(T):.4f} maskPSNR={sum(rows[c][k][4] for c in T)/len(T):.2f} (warped-only maskPSNR {sum(rows[c][k][5] for c in T)/len(T):.2f})  n={len(T)}")
PY
echo "EVAL_GT_DONE $L $(date +%H:%M:%S)"
