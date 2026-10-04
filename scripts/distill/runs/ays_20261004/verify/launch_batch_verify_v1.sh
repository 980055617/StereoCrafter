#!/bin/bash
# ays_20261004 / verify -- launcher: waits for ONE acquisition of /tmp/claude-gpu1.lock and runs the whole GPU batch
# inside it (PREREG.txt "GPU").  Started as a transient systemd user unit (unit name verify-batch-v1).
set -u
cd /home/kawa/master_project/StereoCrafter
L=scripts/distill/runs/ays_20261004/verify
echo "QUEUED $(date '+%F_%T') waiting for /tmp/claude-gpu1.lock" >> $L/batch_verify_v1.log
VERIFY_LOCK=gpu1 flock /tmp/claude-gpu1.lock bash $L/batch_verify_v1.sh
echo "LAUNCHER_EXIT rc=$? $(date '+%F_%T')" >> $L/batch_verify_v1.log
