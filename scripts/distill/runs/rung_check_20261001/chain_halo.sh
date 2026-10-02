#!/bin/bash
# Waits for both render lanes, then runs the all-frames in-hole halo analysis (CPU only) on every
# clip whose mean disocclusion fraction exceeds 0.3 %.
set -u
cd /home/kawa/master_project/StereoCrafter
D=scripts/distill/runs/rung_check_20261001
O=outputs/rung_check_20261001
echo "waiting for render lanes ... $(date +%T)"
until ! systemctl --user is-active --quiet rc-lane-s200 && ! systemctl --user is-active --quiet rc-lane-s400; do sleep 20; done
echo "lanes done, starting halo analysis $(date +%T)"
PY=/home/kawa/miniconda3/envs/stereocrafter/bin/python
export PYTHONPATH=/home/kawa/master_project/StereoCrafter
$PY $D/inhole_halo.py > $O/inhole_halo.log 2>&1
echo "HALO rc=$? $(date +%T)"
echo "CHAIN_HALO_DONE $(date +%T)"
