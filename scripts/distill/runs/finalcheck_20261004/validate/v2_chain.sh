#!/bin/bash
# V2 chain on GPU 1: wait for V1 -> H1 smoke (gate on md5) -> 16 hi-res renders -> scoring.
set -u
cd /home/kawa/master_project/StereoCrafter
V=scripts/distill/runs/finalcheck_20261004/validate
O=outputs/finalcheck_20261004/validate
CL=$O/v2_chain.log
echo "CHAIN_START $(date '+%F %T')" | tee -a $CL
while systemctl --user is-active --quiet fc_validate_v1; do sleep 20; done
echo "V1_UNIT_INACTIVE $(date '+%F %T')" | tee -a $CL
/home/kawa/miniconda3/envs/stereocrafter/bin/python $V/v1_equiv_check.py > $O/v1_temporal/equiv_check.txt 2>&1
EQ=$?; cat $O/v1_temporal/equiv_check.txt | tee -a $CL
if [ $EQ -ne 0 ]; then echo "V1_EQUIV_FAIL -- chain stopped; rerun V1 with the tracked script $(date '+%F %T')" | tee -a $CL; exit 4; fi
bash $V/run_driver_res.sh $V/jobs_smoke.txt $O/control
REF=$(awk '/_sbs\./{print $1}' outputs/beyond_distil_mamba_scaled/clips/0042_mstudent2_step800_deliv_ll/writer_md5.txt)
GOT=$(awk '/_sbs\./{print $1}' $O/control/0042_deliv_ll_576x1024/writer_md5.txt 2>/dev/null)
echo "H1 ref=$REF got=$GOT" | tee -a $CL
if [ -z "$GOT" ] || [ "$REF" != "$GOT" ]; then echo "H1_FAIL -- stopping before the 16 renders $(date '+%F %T')" | tee -a $CL; exit 3; fi
echo "H1_PASS $(date '+%F %T')" | tee -a $CL
bash $V/run_driver_res.sh $V/jobs_hires.txt $O/clips
echo "RENDERS_DONE $(date '+%F %T')" | tee -a $CL
bash $V/v2_score.sh
echo "CHAIN_DONE $(date '+%F %T')" | tee -a $CL
