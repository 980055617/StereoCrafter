#!/bin/bash
# judge J3c: control first; the AYS renders run only if the control reproduces origin_ll 0301's md5.
set -u
cd /home/kawa/master_project/StereoCrafter
J=scripts/distill/runs/more_20261004/judge
bash $J/run_driver_ays_v1.sh $J/jobs_J3c_ays_0301.txt 0
M=$(cut -d' ' -f1 outputs/more_20261004/judge/ays/clips/0301_AC0_origin_g101_T8sig/writer_md5.txt 2>/dev/null | head -1)
if [ "$M" = "2e533d7755c950d2fc95043f6fb0a51d" ]; then
  echo "CONTROL_PASS AC0 md5=$M $(date +%F_%T)"
  bash $J/run_driver_ays_v1.sh $J/jobs_J3c_ays_0301_b.txt 0
else
  echo "CONTROL_FAIL AC0 md5=$M -> STOP (no AYS render) $(date +%F_%T)"
fi
echo "CHAIN_J3C_DONE $(date +%F_%T)"
