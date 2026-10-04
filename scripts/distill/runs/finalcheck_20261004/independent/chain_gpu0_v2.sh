#!/bin/bash
# GPU 0 chain v2: H0 control now -> quality-only T5 renders at 1024x1792 -> wait for R1 -> timing trio at 1024x1792.
set -u
cd /home/kawa/master_project/StereoCrafter
I=scripts/distill/runs/finalcheck_20261004/independent
O=outputs/finalcheck_20261004/independent
LOG=$O/chain_gpu0.log
echo "CHAIN0v2_START $(date +%F_%T)" >> $LOG
$I/run_driver_hres_v1.sh $I/jobs_H0.txt 0
REF=$(grep _sbs outputs/finalcheck_20261004/speed/clips/0042_deliv_g100_T5pad/writer_md5.txt | cut -d' ' -f1)
GOT=$(grep _sbs $O/hres/clips/0042_deliv_g100_T5pad_cfgctl/writer_md5.txt 2>/dev/null | cut -d' ' -f1)
if [ -n "$GOT" ] && [ "$REF" = "$GOT" ]; then echo "H0 PASS got=$GOT ref=$REF $(date +%F_%T)" > $O/H0_RESULT.txt
else echo "H0 FAIL got=${GOT:-none} ref=$REF $(date +%F_%T)" > $O/H0_RESULT.txt; cat $O/H0_RESULT.txt >> $LOG; exit 5; fi
cat $O/H0_RESULT.txt >> $LOG
$I/run_driver_hres_v1.sh $I/jobs_H_1792_quality.txt 0
until grep -q '^SCORE_DONE' $O/SCORES_R1_576.txt 2>/dev/null; do sleep 20; done
echo "R1 seen done $(date +%F_%T)" >> $LOG
$I/run_driver_hres_v1.sh $I/jobs_H_1792_timing.txt 0
echo "CHAIN0_DONE $(date +%F_%T)" >> $LOG
