#!/bin/bash
# DRIVER-EQUIVALENCE CONTROL.  run_driver_rc.sh is a copy of run_driver_v3.sh with a new OUT root
# and SK_UNET pinned instead of inherited.  To prove the copy is functionally identical, re-render
# a DEV clip (0040) that the published dev ladder already rendered with the same rung, and compare
# the PRE-ENCODE writer_md5 digest.  Equal digests = the two drivers produce the same pixels.
set -u
cd /home/kawa/master_project/StereoCrafter
D=scripts/distill/runs/rung_check_20261001
O=outputs/rung_check_20261001
until ! systemctl --user is-active --quiet rc-chain-score; do sleep 20; done
bash $D/run_driver_rc.sh $D/jobs_equiv.txt 0
NEW=$(grep sbs $O/clips/0040_mstudent2_step200_ll/writer_md5.txt | awk '{print $1}')
OLD=$(grep sbs outputs/beyond_distil_mamba_scaled/clips/0040_mstudent2_step200_ll/writer_md5.txt | awk '{print $1}')
{
echo "DRIVER EQUIVALENCE CONTROL -- dev clip 0040, rung step200"
echo "  published dev-ladder render (run_driver_v3.sh): $OLD"
echo "  this run's driver            (run_driver_rc.sh): $NEW"
if [ "$NEW" = "$OLD" ]; then echo "  RESULT: IDENTICAL pre-encode pixels -- the driver copy is equivalent."
else echo "  RESULT: *** DIGESTS DIFFER *** the driver copy is NOT equivalent."; fi
} | tee $O/DRIVER_EQUIVALENCE.txt
echo "CHAIN_EQUIV_DONE $(date +%T)"
