#!/bin/bash
# GPU 0 chain: wait for R1 -> R2 (rescore validate's 16 hi-res renders) -> H0 control -> H1/H2 renders at 1024x1792
set -u
cd /home/kawa/master_project/StereoCrafter
I=scripts/distill/runs/finalcheck_20261004/independent
O=outputs/finalcheck_20261004/independent
LOG=$O/chain_gpu0.log
echo "CHAIN0_START $(date +%F_%T)" >> $LOG
until grep -q '^SCORE_DONE' $O/SCORES_R1_576.txt 2>/dev/null; do sleep 20; done
echo "R1 seen done $(date +%F_%T)" >> $LOG
$I/score_run.sh $O/SCORES_R2_hires_validate.txt $I/scorelist_R2.txt 0
echo "R2 done $(date +%F_%T) $(tail -1 $O/SCORES_R2_hires_validate.txt)" >> $LOG
$I/run_driver_hres_v1.sh $I/jobs_H0.txt 0
REF=$(grep _sbs outputs/finalcheck_20261004/speed/clips/0042_deliv_g100_T5pad/writer_md5.txt | cut -d' ' -f1)
GOT=$(grep _sbs $O/hres/clips/0042_deliv_g100_T5pad_cfgctl/writer_md5.txt 2>/dev/null | cut -d' ' -f1)
if [ -n "$GOT" ] && [ "$REF" = "$GOT" ]; then echo "H0 PASS got=$GOT ref=$REF $(date +%F_%T)" > $O/H0_RESULT.txt
else echo "H0 FAIL got=${GOT:-none} ref=$REF $(date +%F_%T)" > $O/H0_RESULT.txt; cat $O/H0_RESULT.txt >> $LOG; exit 5; fi
cat $O/H0_RESULT.txt >> $LOG
$I/run_driver_hres_v1.sh $I/jobs_H_1792.txt 0
echo "CHAIN0_DONE $(date +%F_%T)" >> $LOG
