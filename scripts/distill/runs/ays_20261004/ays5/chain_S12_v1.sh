#!/bin/bash
# ays_20261004/ays5 chain S1+S2 (PREREG.txt): C0 control -> [C0b + fallback] -> 8 AYS5 origin renders -> pairing/config
# check -> 12-clip scoring (tag S2) -> analysis -> chain_S3_v1.sh (which itself exits unless the claim holds cleanly).
# GPU 0 only; every GPU process runs under flock /tmp/claude-gpu0.lock (inside the driver / the scoring script).
set -u
set +u; source ~/miniconda3/etc/profile.d/conda.sh; conda activate stereocrafter; set -u
cd /home/kawa/master_project/StereoCrafter
L=scripts/distill/runs/ays_20261004/ays5
O=outputs/ays_20261004/ays5
CK=/mnt/ssd_data/stereocrafter_weights/_distill_injected/mamba5slot_plus_stepdistil_up3_train10clip_step800_20261001.pt
echo "CHAIN_S12_START $(date +%F_%T) ckpt_md5=$(md5sum $CK | cut -d' ' -f1)"
[ -e $L/MODE_S1.txt ] && { echo "MODE_S1.txt exists -> refusing to continue (never overwrite)"; exit 3; }
bash $L/run_driver_ays5_v1.sh $L/jobs_C0_control_0204.txt 0
M=$(cut -d' ' -f1 $O/clips/0204_AYS5pad8_origin_g100/writer_md5.txt 2>/dev/null | head -1)
if [ "$M" = "768339b8be14069225a409321b4c5834" ]; then
  echo "CONTROL_C0_PASS 0204 md5=$M -> MODE=reuse $(date +%F_%T)"; echo reuse > $L/MODE_S1.txt
else
  echo "CONTROL_C0_FAIL 0204 md5=$M (want 768339b8be14069225a409321b4c5834) -> C0b on 0301 $(date +%F_%T)"
  bash $L/run_driver_ays5_v1.sh $L/jobs_C0b_control_0301.txt 0
  M2=$(cut -d' ' -f1 $O/clips/0301_AYS5pad8_origin_g100/writer_md5.txt 2>/dev/null | head -1)
  if [ "$M2" = "c00eb301e7e4becfa7ffea30f852527d" ]; then
    echo "CONTROL_C0b_PASS 0301 md5=$M2 -> cross-GPU difference only -> MODE=own, re-render 0052 0147 on GPU 0 $(date +%F_%T)"
    echo own > $L/MODE_S1.txt
    bash $L/run_driver_ays5_v1.sh $L/jobs_S1_fallback_regime.txt 0
  else
    echo "CONTROL_C0b_FAIL 0301 md5=$M2 (want c00eb301e7e4becfa7ffea30f852527d) -> setup broken, STOP $(date +%F_%T)"
    echo "CHAIN_S12_STOPPED"; exit 5
  fi
fi
bash $L/run_driver_ays5_v1.sh $L/jobs_S1_ays5_8clip.txt 0
python $L/check_pairing_ays5_v1.py S1 $L/PAIRING_S1.txt || { echo "PAIRING_S1_FAIL -> STOP before scoring $(date +%F_%T)"; echo "CHAIN_S12_STOPPED"; exit 6; }
echo "PAIRING_S1_PASS $(date +%F_%T)"
bash $L/score_ays5_v1.sh S2 S2 0042 0052 0125 0128 0141 0147 0170 0204 0225 0251 0259 0301
python $L/analyze_ays5_v1.py $L/TABLE_S2_AYS5_12CLIP.txt $L/TABLE_S2_AYS5_12CLIP.json
echo "STEP2_DONE $(date +%F_%T) ckpt_md5=$(md5sum $CK | cut -d' ' -f1)"
bash $L/chain_S3_v1.sh
echo "CHAIN_S12_DONE $(date +%F_%T) ckpt_md5=$(md5sum $CK | cut -d' ' -f1)"
