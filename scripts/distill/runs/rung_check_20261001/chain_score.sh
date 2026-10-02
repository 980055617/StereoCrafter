#!/bin/bash
# Waits for both render lanes, then scores all 6 configs x 12 clips on the two GPUs and builds
# the Task 1 table.  Every reused row is RE-SCORED so the published means act as a harness anchor.
set -u
cd /home/kawa/master_project/StereoCrafter
D=scripts/distill/runs/rung_check_20261001
O=outputs/rung_check_20261001
echo "waiting for render lanes ... $(date +%T)"
until ! systemctl --user is-active --quiet rc-lane-s200 && ! systemctl --user is-active --quiet rc-lane-s400; do sleep 20; done
echo "lanes done $(date +%T)"
NS=$(grep -c "^RUN" $O/timing_gpu0.txt; grep -c "^RUN" $O/timing_gpu1.txt)
echo "RUN lines: $NS"
PY=/home/kawa/miniconda3/envs/stereocrafter/bin/python
export PYTHONPATH=/home/kawa/master_project/StereoCrafter
CUDA_VISIBLE_DEVICES=0 $PY $D/score_rungs.py $O/scores_laneA.txt 0042 0052 0125 0128 0141 0147 > $O/score_laneA.log 2>&1 &
PA=$!
CUDA_VISIBLE_DEVICES=1 $PY $D/score_rungs.py $O/scores_laneB.txt 0170 0204 0225 0251 0259 0301 > $O/score_laneB.log 2>&1 &
PB=$!
wait $PA; RA=$?
wait $PB; RB=$?
echo "score rc: laneA=$RA laneB=$RB $(date +%T)"
$PY $D/table_rungs.py > $O/table_rungs.log 2>&1
echo "TABLE rc=$? $(date +%T)"
echo "CHAIN_SCORE_DONE $(date +%T)"
