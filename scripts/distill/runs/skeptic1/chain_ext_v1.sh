#!/bin/bash
# Wait for the stacking lanes, then run the 12-clip lossless extension (the other 8 clips):
# origin+s25 and origin+student, lossless, so the project finally has a 12-clip codec-clean number
# for BOTH knobs.  origin_ll for all 12 clips already exists in beyond4.
set -u; cd /home/kawa/master_project/StereoCrafter
D=scripts/distill/runs/skeptic1
until ! systemctl --user is-active --quiet sk-gpu0 && ! systemctl --user is-active --quiet sk-gpu1 \
      && ! systemctl --user is-active --quiet sk-g125; do sleep 20; done
echo "lanes free $(date +%T)"
systemd-run --user --collect --unit=sk-ext0 -p Environment=PATH=$PATH bash $PWD/$D/stack_driver_v2.sh $PWD/$D/jobs_ext_gpu0.txt 0
systemd-run --user --collect --unit=sk-ext1 -p Environment=PATH=$PATH bash $PWD/$D/stack_driver_v2.sh $PWD/$D/jobs_ext_gpu1.txt 1
echo "ext lanes launched $(date +%T)"
