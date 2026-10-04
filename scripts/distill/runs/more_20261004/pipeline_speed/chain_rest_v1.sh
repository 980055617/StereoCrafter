#!/bin/bash
# after chain_576_v1: the 576 identity-fix renders (C0 pre-check, C1a/C1b, A1r/A2r repeats, C3 origin, C1w worker),
# then the 1792 chain.  Stops before the 576 C-renders if the C0 2-window pre-check md5 differs from s0's.
set -u
cd /home/kawa/master_project/StereoCrafter
L=scripts/distill/runs/more_20261004/pipeline_speed
O=outputs/more_20261004/pipeline_speed
head -4 $L/jobs_C_576.txt > $L/jobs_C0_576.txt
tail -n +5 $L/jobs_C_576.txt > $L/jobs_C1_576.txt
bash $L/run_driver_stage_v2.sh $L/jobs_C0_576.txt $O/c0_576
M=$(cut -d' ' -f1 $O/c0_576/clips/0301_C0_deliv_T5_576_idfix_2win/writer_md5.txt 2>/dev/null | head -1)
if [ "$M" = "719d6ceb1d9c0d083e4e43754a4ec3e9" ]; then
  echo "C0_PRECHECK_PASS md5=$M $(date +%T)"
  bash $L/run_driver_stage_v2.sh $L/jobs_C1_576.txt $O/c_576
else
  echo "C0_PRECHECK_FAIL md5=$M (expected 719d6ceb...) -- C1 renders NOT run $(date +%T)"
fi
bash $L/chain_1792_v1.sh
echo CHAIN_REST_DONE $(date +%F_%T)
