#!/bin/bash
# ays_20261004/ays5 chain S3 (PREREG.txt STEP 3).  Runs ONLY if TABLE_S2_AYS5_12CLIP.json says the step-2 claim HOLDS
# with all 12 S2 reproduction gates passing and no VOID clip; otherwise it exits without any GPU work.
# 3a: origin AYS8 @1.00 on 12 clips; 3b: deliverable AYS5 @1.00 pad8 on 4 regime clips; pairing; scoring (tag S3); analysis.
set -u
set +u; source ~/miniconda3/etc/profile.d/conda.sh; conda activate stereocrafter; set -u
cd /home/kawa/master_project/StereoCrafter
L=scripts/distill/runs/ays_20261004/ays5
CK=/mnt/ssd_data/stereocrafter_weights/_distill_injected/mamba5slot_plus_stepdistil_up3_train10clip_step800_20261001.pt
python - <<'PY' || { echo "S3_NOT_RUN: precondition not met (claim does not hold, a gate failed, or a clip is VOID) $(date +%F_%T)"; exit 0; }
import json, sys
j = json.load(open("scripts/distill/runs/ays_20261004/ays5/TABLE_S2_AYS5_12CLIP.json"))
v = j.get("verdict_step2", {})
g = [x for k, x in j["gates"].items() if k.startswith("S2_")]
ok = bool(v.get("holds")) and len(g) == 12 and all(x["ok"] for x in g) and not v.get("void")
print(f"S3 precondition: holds={v.get('holds')} gates_ok={sum(x['ok'] for x in g)}/{len(g)} void={v.get('void')} -> {'RUN' if ok else 'NOT RUN'}")
sys.exit(0 if ok else 1)
PY
echo "CHAIN_S3_START $(date +%F_%T) ckpt_md5=$(md5sum $CK | cut -d' ' -f1)"
bash $L/run_driver_ays5_v1.sh $L/jobs_S3a_ays8g100_12clip.txt 0
bash $L/run_driver_ays5_v1.sh $L/jobs_S3b_delivays5_4clip.txt 0
python $L/check_pairing_ays5_v1.py S3 $L/PAIRING_S3.txt || { echo "PAIRING_S3_FAIL -> STOP before scoring $(date +%F_%T)"; echo "CHAIN_S3_STOPPED"; exit 6; }
echo "PAIRING_S3_PASS $(date +%F_%T)"
bash $L/score_ays5_v1.sh S3 S3 0042 0052 0125 0128 0141 0147 0170 0204 0225 0251 0259 0301
python $L/analyze_ays5_v1.py $L/TABLE_S3_AYS5_FULL.txt $L/TABLE_S3_AYS5_FULL.json
echo "CHAIN_S3_DONE $(date +%F_%T) ckpt_md5=$(md5sum $CK | cut -d' ' -f1)"
