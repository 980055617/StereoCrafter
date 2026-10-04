#!/bin/bash
# Hi-res scoring: 1024x1792 list once GPU 0's renders (incl. H4) are done; 1024x1920 list once GPU 1's are done. GPU 0.
set -u
cd /home/kawa/master_project/StereoCrafter
I=scripts/distill/runs/finalcheck_20261004/independent
O=outputs/finalcheck_20261004/independent
PY=/home/kawa/miniconda3/envs/stereocrafter/bin/python
LOG=$O/chain_score_hires.log
echo "SCORECHAIN_START $(date +%F_%T)" >> $LOG
N=0; until grep -q '^H4_DONE gpu0' $O/chain_gpu0.log 2>/dev/null; do sleep 20; N=$((N+1)); [ $N -gt 540 ] && { echo "gave up waiting gpu0" >> $LOG; exit 6; }; done
$I/score_run.sh $O/SCORES_HIRES_1792.txt $I/scorelist_HIRES_1792.txt 0
echo "1792 scored $(date +%F_%T) $(tail -1 $O/SCORES_HIRES_1792.txt)" >> $LOG
CUDA_VISIBLE_DEVICES= $PY $I/p3_check.py $O/P3_PROVENANCE_HIRES_1792.txt $O/P3_PROVENANCE_HIRES_1792.json $I/scorelist_HIRES_1792.txt > /dev/null 2>&1
echo "P3 1792: $(tail -1 $O/P3_PROVENANCE_HIRES_1792.txt)" >> $LOG
N=0; until grep -q '^H4_DONE gpu1' $O/chain_gpu1.log 2>/dev/null; do sleep 20; N=$((N+1)); [ $N -gt 540 ] && { echo "gave up waiting gpu1" >> $LOG; exit 6; }; done
$I/score_run.sh $O/SCORES_HIRES_1920.txt $I/scorelist_HIRES_1920.txt 0
echo "1920 scored $(date +%F_%T) $(tail -1 $O/SCORES_HIRES_1920.txt)" >> $LOG
CUDA_VISIBLE_DEVICES= $PY $I/p3_check.py $O/P3_PROVENANCE_HIRES_1920.txt $O/P3_PROVENANCE_HIRES_1920.json $I/scorelist_HIRES_1920.txt > /dev/null 2>&1
echo "P3 1920: $(tail -1 $O/P3_PROVENANCE_HIRES_1920.txt)" >> $LOG
echo "SCORECHAIN_DONE $(date +%F_%T)" >> $LOG
