#!/bin/bash
# ays_20261004 / robust -- PREREG (1b) AYS5 registered pass.  Polls (every 60 s) until the 12 AYS5 origin renders are
# complete and paired (build_rows_ays5_v1.py exit 0), or until the CUTOFF 2026-10-04 22:30 JST -> "NOT RUN".
# Then the same registered scoring as pass 1: score_registered_v1.py (unchanged) on 12 clips + v1w on 0125, GPU 1 under
# /tmp/claude-gpu1.lock, one process per clip.  Never overwrites.
set -u
cd /home/kawa/master_project/StereoCrafter
set +u; source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate stereocrafter; set -u
R=scripts/distill/runs/ays_20261004/robust
O=outputs/ays_20261004/robust
ER=scripts/distill/runs/more_20261004/eval_robustness
LOG=$R/chain_ays5_v1.log
ROWS5=$R/ROWS_ays5.json
CHK=$R/AYS5_READINESS_PAIRING.txt
CLIPS="0042 0052 0125 0128 0141 0147 0170 0204 0225 0251 0259 0301"
CUTOFF=$(date -d '2026-10-04 22:30:00' +%s)
echo "CHAIN_AYS5_START $(date '+%F_%T') cutoff 2026-10-04_22:30:00" >> $LOG
md5sum -c >> $LOG 2>&1 <<EOF || { echo "MD5 GUARD FAILED -- stop" >> $LOG; exit 3; }
c9532d5a5cde42f7024a2caebb855c3f  $ER/score_registered_v1.py
a5a594e9c2df9d42516771f236428556  $ER/score_registered_v1w.py
EOF
LAST=""
while :; do
  MSG=$(python $R/build_rows_ays5_v1.py $ROWS5 $CHK 2>&1); RC=$?
  if [ "$MSG" != "$LAST" ]; then echo "POLL $(date '+%T') rc=$RC :: $(echo "$MSG" | tail -1)" >> $LOG; LAST=$MSG; fi
  if [ $RC -eq 0 ]; then echo "$MSG" >> $LOG; break; fi
  if [ $RC -eq 20 ]; then echo "AYS5_PASS_NOT_RUN pairing/config FAIL (see $CHK) $(date '+%F_%T')" >> $LOG; exit 0; fi
  if [ $RC -ne 10 ]; then echo "AYS5_PASS_NOT_RUN builder error rc=$RC: $MSG" >> $LOG; exit 0; fi
  if [ "$(date +%s)" -ge "$CUTOFF" ]; then
    echo "AYS5_PASS_NOT_RUN 12 complete paired AYS5 renders not on disk by the cutoff (last: $(echo "$MSG" | tail -1)) $(date '+%F_%T')" >> $LOG
    exit 0
  fi
  sleep 60
done
O2=$O/score_reg_ays5_v1
O2W=$O/score_reg_ays5_v1_wide
if [ -e "$O2" ] || [ -e "$O2W" ]; then echo "REFUSING: $O2 or $O2W exists" >> $LOG; exit 1; fi
mkdir -p "$O2" "$O2W"
echo "AYS5_SCORING_START $(date '+%F_%T')" >> $LOG
for c in $CLIPS; do
  CUDA_VISIBLE_DEVICES=1 SCORE_STEP=4 flock /tmp/claude-gpu1.lock \
    python $ER/score_registered_v1.py "$O2" $ROWS5 "$c" >> $R/score_reg_ays5_v1.log 2>&1
  echo "REG5_CLIP_RC $c rc=$? $(date '+%F_%T')" >> $LOG
done
CUDA_VISIBLE_DEVICES=1 SCORE_STEP=4 REG_DDY=-20,20 REG_DDX=-240,40 flock /tmp/claude-gpu1.lock \
  python $ER/score_registered_v1w.py "$O2W" $ROWS5 0125 >> $R/score_reg_ays5_v1_wide.log 2>&1
echo "REG5_WIDE_RC 0125 rc=$? $(date '+%F_%T')" >> $LOG
echo "CHAIN_AYS5_END $(date '+%F_%T')" >> $LOG
