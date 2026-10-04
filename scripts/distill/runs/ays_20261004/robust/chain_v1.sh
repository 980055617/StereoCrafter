#!/bin/bash
# ays_20261004 / robust lane chain (PREREG.txt).  GPU 1 only, every GPU command under /tmp/claude-gpu1.lock.
#   stage 1  registered-GT pass 1: score_registered_v1.py (UNCHANGED, run from its own path) on 12 clips, then
#            score_registered_v1w.py on 0125 with the widened grid (inherited eval_robustness addendum)
#   stage 2  temporal: score_temporal_ll.py (UNCHANGED, validate lane) one process per clip, origin first
# Never overwrites: refuses existing output dirs/files.  Rows/paths come from ROWS_pass1.json (collect_rows_v1.py).
set -u
cd /home/kawa/master_project/StereoCrafter
set +u; source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate stereocrafter; set -u
R=scripts/distill/runs/ays_20261004/robust
O=outputs/ays_20261004/robust
ER=scripts/distill/runs/more_20261004/eval_robustness
VAL=scripts/distill/runs/finalcheck_20261004/validate
LOG=$R/chain_v1.log
ROWS=$R/ROWS_pass1.json
CLIPS="0042 0052 0125 0128 0141 0147 0170 0204 0225 0251 0259 0301"
echo "CHAIN_V1_START $(date '+%F_%T') python=$(which python) host=$(hostname)" >> $LOG
md5sum -c >> $LOG 2>&1 <<EOF || { echo "MD5 GUARD FAILED -- stop" >> $LOG; exit 3; }
c9532d5a5cde42f7024a2caebb855c3f  $ER/score_registered_v1.py
a5a594e9c2df9d42516771f236428556  $ER/score_registered_v1w.py
24bf1bcbdf11b454038f89bd96f70701  $VAL/score_temporal_ll.py
EOF

# ------------------------------------------------------------------ stage 1: registered pass 1
O1=$O/score_reg_v1
O1W=$O/score_reg_v1_wide
if [ -e "$O1" ] || [ -e "$O1W" ]; then
  echo "STAGE1 SKIPPED: $O1 or $O1W exists (never overwritten)" >> $LOG
else
  mkdir -p "$O1" "$O1W"
  for c in $CLIPS; do
    CUDA_VISIBLE_DEVICES=1 SCORE_STEP=4 flock /tmp/claude-gpu1.lock \
      python $ER/score_registered_v1.py "$O1" $ROWS "$c" >> $R/score_reg_v1.log 2>&1
    echo "REG_CLIP_RC $c rc=$? $(date '+%F_%T')" >> $LOG
  done
  CUDA_VISIBLE_DEVICES=1 SCORE_STEP=4 REG_DDY=-20,20 REG_DDX=-240,40 flock /tmp/claude-gpu1.lock \
    python $ER/score_registered_v1w.py "$O1W" $ROWS 0125 >> $R/score_reg_v1_wide.log 2>&1
  echo "REG_WIDE_RC 0125 rc=$? $(date '+%F_%T')" >> $LOG
fi

# ------------------------------------------------------------------ stage 2: temporal (every frame, STEP unset)
OT=$O/temporal_v1
mkdir -p "$OT"
for c in $CLIPS; do
  OUTJ=$OT/$c.json
  if [ -e "$OUTJ" ]; then echo "TEMPORAL SKIP $c: $OUTJ exists (never overwritten)" >> $LOG; continue; fi
  SPECS=$(python - "$ROWS" "$c" <<'PY'
import json, sys
d = json.load(open(sys.argv[1])); c = sys.argv[2]
order = ["origin_ll", "AYS8_origin_g101", "mstudent2_step800_deliv_ll", "deliv_g100_T5nat", "deliv_g100_T5pad",
         "origin_g100_T5pad"]
print(" ".join(f"{c}={d['cells'][c][lab]['path']}" for lab in order))
PY
)
  echo "TEMPORAL_START $c $(date '+%F_%T') specs: $SPECS" >> $LOG
  env -u STEP -u NOFLOW -u GT_DIR CUDA_VISIBLE_DEVICES=1 flock /tmp/claude-gpu1.lock \
    python $VAL/score_temporal_ll.py "$OUTJ" $SPECS > $OT/$c.log 2>&1
  RC=$?
  RAFT=$(grep -c "^\[temporal\] RAFT loaded" $OT/$c.log)
  echo "TEMPORAL_RC $c rc=$RC raft_loaded=$RAFT $(date '+%F_%T')" >> $LOG
done
echo "CHAIN_V1_END $(date '+%F_%T')" >> $LOG
