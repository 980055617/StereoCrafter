#!/bin/bash
# Stripes lane chain: K1 gate -> smoke (0301) -> integrity gate -> regime set.  GPU lock is taken PER JOB inside
# run_driver_stripes_v1.sh; this chain itself holds no lock.  Stops at the first failed gate.
set -u
cd /home/kawa/master_project/StereoCrafter
L=scripts/distill/runs/more_20261004/stripes
O=outputs/more_20261004/stripes
PY=/home/kawa/miniconda3/envs/stereocrafter/bin/python
log() { echo "CHAIN $(date +%F_%T) $*" | tee -a $O/chain.log; }
log "start; waiting for unit stripes-k1-control"
while systemctl --user is-active -q stripes-k1-control; do sleep 10; done
MD5=$(cut -d' ' -f1 $O/clips/0301_deliv_g100_T5nat_K1none/writer_md5.txt 2>/dev/null | head -1)
if [ "$MD5" != "a41b432baeb25393bb12d39b4f2e6a21" ]; then log "K1 FAIL md5=$MD5 -- stopping"; exit 1; fi
log "K1 PASS md5=$MD5"
$L/run_driver_stripes_v1.sh $L/jobs_00b_smoke.txt 0 > $O/driver_00b_smoke.log 2>&1
CUDA_VISIBLE_DEVICES= $PY $L/check_k2_k4_v1.py $O/CHECK_K1K2K4_after_smoke.txt > /dev/null 2>&1
if ! grep -q "SUMMARY renders=7 failed_checks=0" $O/CHECK_K1K2K4_after_smoke.txt; then
  log "smoke integrity FAIL -- stopping: $(tail -1 $O/CHECK_K1K2K4_after_smoke.txt)"; exit 2; fi
log "smoke integrity PASS: $(tail -1 $O/CHECK_K1K2K4_after_smoke.txt)"
$L/run_driver_stripes_v1.sh $L/jobs_01_regime.txt 0 > $O/driver_01_regime.log 2>&1
CUDA_VISIBLE_DEVICES= $PY $L/check_k2_k4_v1.py $O/CHECK_K1K2K4_after_regime.txt > /dev/null 2>&1
log "regime done: $(tail -1 $O/CHECK_K1K2K4_after_regime.txt)"
