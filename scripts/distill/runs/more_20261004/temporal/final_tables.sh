#!/bin/bash
# Build the final 4-clip table (CPU only).  Existing speed-lane LPIPS files are passed FIRST so this lane's rescored
# origin/BASE rows (identical by S2) win and the deliv8/BASEpad context rows appear.  Temporal JSONs: t0 first, then the
# invocations that contain the candidates (per clip: 0170 from t1 + t3, the others from t2_<clip>).
set -u
cd /home/kawa/master_project/StereoCrafter
T=scripts/distill/runs/more_20261004/temporal
S=outputs/more_20261004/temporal/scores
SP=scripts/distill/runs/finalcheck_20261004/speed
python3 $T/analyze_cand_v1.py $T/TABLE_TEMPORAL_4CLIP.txt --clips 0170 0259 0042 0301 \
  --cands T5nat_ov5 T5nat_ov7 T5nat_pw03 T5nat_pw05 T5nat_nsfa T5nat_dcs14 --ctx origin_g101_s8_dcs14 \
  --tjson $S/t0_baselines.json $S/t1_smoke0170.json $S/t3_0170_ctx.json $S/t2_0259.json $S/t2_0042.json $S/t2_0301.json \
  --lpips $SP/SCORES_STEP1.txt $SP/SCORES_STEP2_EXT12.txt $SP/SCORES_STEP2_T5G100_4clip.txt $SP/SCORES_POSTHOC_T5G100NAT.txt \
          $T/SCORES_LPIPS_T1_SMOKE0170.txt $T/SCORES_LPIPS_T3_0170CTX.txt $T/SCORES_LPIPS_T2_0259.txt \
          $T/SCORES_LPIPS_T2_0042.txt $T/SCORES_LPIPS_T2_0301.txt
