#!/bin/bash
# ays_20261004/ays5 OPTIONAL registered-GT rescore (PREREG.txt "OPTIONAL"; gates nothing).  Derived from
# scripts/distill/runs/more_20261004/eval_robustness/run_score_v1.sh: the scorer score_registered_v1.py is run UNCHANGED
# (read-only, from its lane dir), one process per clip on GPU 0 under /tmp/claude-gpu0.lock, rows JSON + output in this
# lane.  The output dir must not exist yet (never overwrite).
set -u
set +u; source ~/miniconda3/etc/profile.d/conda.sh; conda activate stereocrafter; set -u
cd /home/kawa/master_project/StereoCrafter
L=scripts/distill/runs/ays_20261004/ays5
E=scripts/distill/runs/more_20261004/eval_robustness
O=outputs/ays_20261004/ays5/registered_v1
LOG=$L/registered_v1.log
if [ -e "$O" ]; then echo "REFUSING: $O exists" >> "$LOG"; exit 1; fi
python $L/build_rows_registered_v1.py $L/ROWS_registered_v1.json >> "$LOG" 2>&1 || { echo "ROWS_BUILD_FAIL" >> "$LOG"; exit 2; }
mkdir -p "$O"
echo "REG_START $(date '+%F_%T') out=$O scorer_md5=$(md5sum $E/score_registered_v1.py | cut -d' ' -f1)" >> "$LOG"
for c in 0042 0052 0125 0128 0141 0147 0170 0204 0225 0251 0259 0301; do
  CUDA_VISIBLE_DEVICES=0 SCORE_STEP=4 flock /tmp/claude-gpu0.lock \
    python $E/score_registered_v1.py "$O" $L/ROWS_registered_v1.json "$c" >> "$LOG" 2>&1
  echo "CLIP_RC $c rc=$? $(date '+%F_%T')" >> "$LOG"
done
python $L/analyze_registered_v1.py $L/TABLE_REGISTERED_v1.txt >> "$LOG" 2>&1
echo "REG_END $(date '+%F_%T')" >> "$LOG"
