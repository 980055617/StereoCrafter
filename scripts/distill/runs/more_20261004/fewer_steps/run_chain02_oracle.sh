#!/bin/bash
# chain 02: wait for chain 01 to finish, then O-C2 (no substitution) and O1 (the oracle), each job under the GPU-0 lock
set -u
cd /home/kawa/master_project/StereoCrafter
L=scripts/distill/runs/more_20261004/fewer_steps
until ! systemctl --user is-active --quiet fs-chain01-smoke0301; do sleep 20; done
unset OR_SUBST OR_M OR_PREFIX OR_E2 OR_CHECK
FS_RUN=$L/oracle_fs_v1.py $L/run_driver_fs_v1.sh $L/jobs_02a_oracleC2.txt 0
FS_RUN=$L/oracle_fs_v1.py OR_SUBST=1,2 OR_M=1:4,2:8 OR_PREFIX=2:4:0.09738767892122269 OR_E2=2:0.09738767892122269 OR_CHECK=1 \
  $L/run_driver_fs_v1.sh $L/jobs_02b_oracleO1.txt 0
echo CHAIN02_DONE
