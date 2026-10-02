#!/bin/bash
# SLOT BUDGET runner: pass A (published bench2.py totals) then pass B (per-slot attribution).
# Sequential so the GPU stays exclusive for both, exactly as the published bench required.
set -u
cd /home/kawa/master_project/StereoCrafter
R=scripts/distill/runs/slotbudget
bash $R/bench_totals_v1.sh
bash $R/bench_slots_v1.sh
echo "SLOTBUDGET_ALL_DONE $(date +%F_%H:%M:%S)"
