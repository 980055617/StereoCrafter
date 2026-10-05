#!/bin/bash
# S0: K1 control, md5 gate, then the 8 smoke variant renders on 0301.  GPU lock is taken per job inside the driver.
set -u
cd /home/kawa/master_project/StereoCrafter
L=scripts/distill/runs/deep_20261004/sdedit
O=outputs/deep_20261004/sdedit
echo "CHAIN00 start $(date +%F_%T)"
bash $L/run_driver_sdedit_v1.sh $L/jobs_00a_k1.txt 0
GOT=$(grep "_sbs" $O/clips/0301_origin_g100_s8_k1ctl/writer_md5.txt 2>/dev/null | cut -d' ' -f1)
REF=486dcefba769a992be0342f1f5eea349
if [ "$GOT" = "$REF" ]; then echo "K1 PASS writer md5 $GOT == $REF"; else echo "K1 FAIL writer md5 '$GOT' != $REF -- stopping"; exit 2; fi
bash $L/run_driver_sdedit_v1.sh $L/jobs_00b_smoke.txt 0
echo "CHAIN00 done $(date +%F_%T)"
