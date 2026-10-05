#!/bin/bash
# decoder_ft GATE G0 (scorer-chain reproduction, no new information): on test clip 0301 the stock re-decode of the captured
# origin latents (byte-identical to the published origin_ll render, gate C2) is scored through this lane's whole chain
# (score_clip_ll -> make_rows -> score_registered_df_v1 -> score_aux_v1).  PASS iff UNREG == the published origin_ll row
# (ays_20261004/robust/ROWS_pass1.json) and REG_FRAME/REG_CLIP == eval_robustness score_v1/0301.json origin_ll, |d| <= 1e-6.
# usage: bash chain_g0_v1.sh <gpu>
set -u
cd /home/kawa/master_project/StereoCrafter
set +u; source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate stereocrafter; set -u
L=scripts/distill/runs/deep_20261004/decoder_ft
GPU=$1
export CUDA_VISIBLE_DEVICES=$GPU
P=/mnt/ssd_data/deep_20261004/decoder_ft/gate_stock/0301_origin_cap__stock/0301_inpainting_results_sbs.mkv
SCORE_STEP=4 flock /tmp/claude-gpu$GPU.lock python scripts/distill/runs/fulldata_v2/beyond4/score_clip_ll.py 0301=$P > $L/g0_unreg_0301.txt 2>&1 < /dev/null
python $L/make_rows_v1.py $L/rows_g0_0301.json origin_cap__stock,origin_ll $L/g0_unreg_0301.txt --base=scripts/distill/runs/ays_20261004/robust/ROWS_pass1.json:origin_ll
O=outputs/deep_20261004/decoder_ft/score_reg_g0
mkdir -p $O
SCORE_STEP=4 flock /tmp/claude-gpu$GPU.lock python $L/score_registered_df_v1.py $O $L/rows_g0_0301.json 0301 > $L/g0_reg_0301.log 2>&1 < /dev/null
A=outputs/deep_20261004/decoder_ft/score_aux_g0
flock /tmp/claude-gpu$GPU.lock env PYTHONPATH=/mnt/ssd_data/deep_20261004/blur_diag/pylib:/mnt/ssd_data/deep_20261004/skeptic/pylib \
  TORCH_HOME=/mnt/ssd_data/deep_20261004/decoder_ft/torch_home HF_HOME=/mnt/ssd_data/deep_20261004/blur_diag/hf_home \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python $L/score_aux_v1.py $A $L/rows_g0_0301.json $O 0301 > $L/g0_aux_0301.log 2>&1 < /dev/null
python - <<'EOF' | tee $L/GATE_G0.txt
import json
R = json.load(open("outputs/deep_20261004/decoder_ft/score_reg_g0/0301.json"))["configs"]
E = json.load(open("outputs/more_20261004/eval_robustness/score_v1/0301.json"))["configs"]["origin_ll"]["lpips_clip"]
pub = json.load(open("scripts/distill/runs/ays_20261004/robust/ROWS_pass1.json"))["cells"]["0301"]["origin_ll"]["lpips"]
ok = True
for v in ("UNREG", "REG_CLIP", "REG_FRAME"):
    a, b, c = R["origin_cap__stock"]["lpips_clip"][v], R["origin_ll"]["lpips_clip"][v], E[v]
    good = abs(a - c) <= 1e-6 and abs(b - c) <= 1e-6
    ok &= good
    print(f"G0 0301 {v:9s} lane re-decode {a:.6f}  published-path {b:.6f}  eval_robustness {c:.6f} -> {'OK' if good else 'MISMATCH'}")
u = R["origin_cap__stock"]["lpips_clip"]["UNREG"]
ok &= abs(u - pub) <= 1e-6
print(f"G0 0301 UNREG vs published ROWS_pass1 {pub:.6f}: |d| {abs(u - pub):.1e}")
aux = json.load(open("outputs/deep_20261004/decoder_ft/score_aux_g0/0301.json"))["labels"]
print("G0 aux: lane re-decode vs published path identical:", aux["origin_cap__stock"]["dists"] == aux["origin_ll"]["dists"]
      and aux["origin_cap__stock"]["niqe"] == aux["origin_ll"]["niqe"])
print(f"GATE G0 {'PASS' if ok else 'FAIL'}")
EOF
echo "G0_DONE $(date +%F_%T)"
