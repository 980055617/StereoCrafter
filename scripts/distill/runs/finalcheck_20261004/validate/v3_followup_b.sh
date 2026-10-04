#!/bin/bash
# [_b: waits for fc_validate_v2chain_b] Follow-up on GPU 1 after the V2 chain: (1) V1 diagnostics D1/D2 (PREREG ADDENDUM 1), (2) optional hi-res temporal.
# Proceeds only if the chain logged CHAIN_DONE.  New files only; nothing is overwritten.
set -u
cd /home/kawa/master_project/StereoCrafter
PY=/home/kawa/miniconda3/envs/stereocrafter/bin/python
V=scripts/distill/runs/finalcheck_20261004/validate
O=outputs/finalcheck_20261004/validate
FL=$O/v3_followup_b.log
export CUDA_VISIBLE_DEVICES=1
unset STEP NOFLOW GT_DIR
echo "FOLLOWUP_START $(date '+%F %T')" | tee -a $FL
while systemctl --user is-active --quiet fc_validate_v2chain_b; do sleep 30; done
if ! grep -q CHAIN_DONE $O/v2_chain_b.log; then echo "CHAIN did not finish (no CHAIN_DONE) -- follow-up not run $(date '+%F %T')" | tee -a $FL; exit 5; fi
# (1) D1/D2
if [ ! -e $O/v1_temporal/diag_D1D2.json ]; then
  /usr/bin/time -v $PY $V/v1_diag.py $O/v1_temporal/temporal_576_12clip.json $O/v1_temporal/diag_D1D2.json > $O/v1_temporal/diag_D1D2.log 2>&1
  echo "DIAG rc=$? $(date '+%F %T')" | tee -a $FL
fi
# (2) hi-res temporal (copy scorer, origin first per clip)
mkdir -p $O/v2_temporal
for res in 1024x1792 1024x1920; do
  J=$O/v2_temporal/temporal_${res}.json
  [ -e $J ] && { echo "SKIP $J exists" | tee -a $FL; continue; }
  ARGS=""
  for c in 0170 0204 0042 0052; do
    for m in origin deliv; do ARGS="$ARGS $c=$O/clips/${c}_${m}_ll_${res}/${c}_inpainting_results_sbs.mkv"; done
  done
  /usr/bin/time -v $PY $V/score_temporal_ll.py $J $ARGS > $O/v2_temporal/temporal_${res}.log 2>&1
  echo "HIRES_TEMPORAL $res rc=$? $(date '+%F %T')" | tee -a $FL
done
echo "FOLLOWUP_DONE $(date '+%F %T')" | tee -a $FL
