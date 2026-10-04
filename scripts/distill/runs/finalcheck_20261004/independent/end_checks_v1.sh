#!/bin/bash
# P1 + P2 end checks (CPU only).  usage: end_checks_v1.sh OUT.txt
set -u
cd /home/kawa/master_project/StereoCrafter
OUT=$1; [ -e "$OUT" ] && { echo "refusing to overwrite $OUT"; exit 3; }
{
echo "END CHECKS $(date '+%F %T')"
echo "--- P1 protected checkpoints (expected 08cf44850b8f392efb307e3a48cd82d1 / e9c232878319d041680e7fb3be74bf10)"
md5sum /mnt/ssd_data/stereocrafter_weights/_distill_injected/mamba5slot_plus_stepdistil_up3_train10clip_step800_20261001.pt \
       /mnt/ssd_data/stereocrafter_weights/_distill_injected/light_lvl0_fulldata333_v2_8k_mamba_only.pt
echo "--- P2 git status --porcelain (expected: ' M docs/agents/model-change-log.md' [pre-existing, mtime 2026-10-03 14:52], '?? scripts/distill/runs/clean_controls/', '?? scripts/distill/runs/finalcheck_20261004/')"
git status --porcelain
echo "--- mtime of the one modified tracked file"
stat -c '%y %n' docs/agents/model-change-log.md
echo "--- tracked scorer / temporal scripts unchanged (md5 at lane start: score_clip_ll.py a6695eb0..., score_temporal.py 4cb3c07d...)"
md5sum scripts/distill/runs/fulldata_v2/beyond4/score_clip_ll.py scripts/distill/score_temporal.py
} > $OUT 2>&1
cat $OUT
