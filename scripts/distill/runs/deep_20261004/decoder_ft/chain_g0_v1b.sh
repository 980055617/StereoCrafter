#!/bin/bash
# decoder_ft GATE G0, second attempt.  chain_g0_v1.sh failed before scoring anything: make_rows_v1.py --base merged the
# published origin_ll rows of ALL 12 clips and then refused because origin_cap__stock exists only for 0301.  Here the rows
# json is built for 0301 only (the ROW line of g0_unreg_0301.txt, already computed, + the published 0301 origin_ll row).
set -u
cd /home/kawa/master_project/StereoCrafter
set +u; source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate stereocrafter; set -u
L=scripts/distill/runs/deep_20261004/decoder_ft
GPU=$1
export CUDA_VISIBLE_DEVICES=$GPU
python - <<'PY'
import json
L = "scripts/distill/runs/deep_20261004/decoder_ft"
row = [l for l in open(f"{L}/g0_unreg_0301.txt") if l.startswith("ROW ")]
assert len(row) == 1
kv = dict(t.split("=", 1) for t in row[0].split()[1:])
mine = dict(dy=int(kv["dy"]), dx=int(kv["dx"]), leftPSNR=float(kv["leftPSNR"]), lpips=float(kv["lpips"]), sharp=float(kv["sharp"]),
            gtSharp=float(kv["gtSharp"]), rightPSNR=float(kv["rightPSNR"]), n=int(kv["n"]), path=kv["path"], sources=[f"{L}/g0_unreg_0301.txt"])
pub = json.load(open("scripts/distill/runs/ays_20261004/robust/ROWS_pass1.json"))["cells"]["0301"]["origin_ll"]
out = f"{L}/rows_g0b_0301.json"
json.dump(dict(labels=["origin_cap__stock", "origin_ll"], cells={"0301": {"origin_cap__stock": mine, "origin_ll": dict(pub, sources=["ays_20261004/robust/ROWS_pass1.json"])}},
               clips=["0301"]), open(out, "w"), indent=1)
print("wrote", out, "UNREG lane", mine["lpips"], "published", pub["lpips"])
PY
O=outputs/deep_20261004/decoder_ft/score_reg_g0b
A=outputs/deep_20261004/decoder_ft/score_aux_g0b
mkdir -p $O $A
SCORE_STEP=4 flock /tmp/claude-gpu$GPU.lock python $L/score_registered_df_v1.py $O $L/rows_g0b_0301.json 0301 > $L/g0b_reg_0301.log 2>&1 < /dev/null
flock /tmp/claude-gpu$GPU.lock env PYTHONPATH=/mnt/ssd_data/deep_20261004/blur_diag/pylib:/mnt/ssd_data/deep_20261004/skeptic/pylib \
  TORCH_HOME=/mnt/ssd_data/deep_20261004/decoder_ft/torch_home HF_HOME=/mnt/ssd_data/deep_20261004/blur_diag/hf_home \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python $L/score_aux_v1.py $A $L/rows_g0b_0301.json $O 0301 > $L/g0b_aux_0301.log 2>&1 < /dev/null
python - <<'EOF2' | tee $L/GATE_G0.txt
import json
R = json.load(open("outputs/deep_20261004/decoder_ft/score_reg_g0b/0301.json"))["configs"]
E = json.load(open("outputs/more_20261004/eval_robustness/score_v1/0301.json"))["configs"]["origin_ll"]["lpips_clip"]
pub = json.load(open("scripts/distill/runs/ays_20261004/robust/ROWS_pass1.json"))["cells"]["0301"]["origin_ll"]["lpips"]
ok = True
for v in ("UNREG", "REG_CLIP", "REG_FRAME", "BLK_LOCAL"):
    a, b, c = R["origin_cap__stock"]["lpips_clip"][v], R["origin_ll"]["lpips_clip"][v], E[v]
    good = abs(a - c) <= 1e-6 and abs(b - c) <= 1e-6
    ok &= good
    print(f"G0 0301 {v:9s} lane re-decode {a:.6f}  published-path {b:.6f}  eval_robustness {c:.6f} -> {'OK' if good else 'MISMATCH'}")
u = R["origin_cap__stock"]["lpips_clip"]["UNREG"]
ok &= abs(u - pub) <= 1e-6
print(f"G0 0301 UNREG vs published ROWS_pass1 {pub:.6f}: |d| {abs(u - pub):.1e}")
aux = json.load(open("outputs/deep_20261004/decoder_ft/score_aux_g0b/0301.json"))["labels"]
same = all(aux["origin_cap__stock"][k] == aux["origin_ll"][k] for k in ("dists", "niqe", "musiq", "lpips_vgg", "psnr_reg"))
ok &= same
print("G0 aux: lane re-decode vs published path identical on dists/niqe/musiq/lpips_vgg/psnr_reg:", same,
      {k: round(aux["origin_cap__stock"][k], 4) for k in ("dists", "niqe", "musiq", "lpips_vgg", "psnr_reg")})
print(f"GATE G0 {'PASS' if ok else 'FAIL'}")
EOF2
echo "G0B_DONE $(date +%F_%T)"
