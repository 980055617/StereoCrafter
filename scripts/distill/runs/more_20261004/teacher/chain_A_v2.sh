#!/bin/bash
# chain A (v2 = v1 + euler_nd GPU test, logs renamed): GPU unit test (bit-exact Euler vs live diffusers + RNG pairing) -> gate render -> 2-window smokes.
set -u
cd /home/kawa/master_project/StereoCrafter
D=scripts/distill/runs/more_20261004/teacher
PY=/home/kawa/miniconda3/envs/stereocrafter/bin/python
echo "CHAIN_A_START $(date +%F_%T)"
CUDA_VISIBLE_DEVICES=1 flock /tmp/claude-gpu1.lock $PY $D/test_teacher_sched_v1.py gpu > $D/test_gpu_v2.log 2>&1
RC=$?
echo "gpu unit test rc=$RC"; grep -E "GPU" $D/test_gpu_v2.log; CUDA_VISIBLE_DEVICES=1 flock /tmp/claude-gpu1.lock $PY $D/test_euler_nd_v1.py gpu 2>&1 | grep -v "FutureWarning\|impl_abstract" > $D/test_euler_nd_gpu_v1.log; cat $D/test_euler_nd_gpu_v1.log
if [ $RC -ne 0 ]; then echo "CHAIN_A_ABORT unit test failed"; exit 1; fi
bash $D/run_driver_teacher_v1.sh $D/jobs_00_gate.txt
G=$(head -1 outputs/more_20261004/teacher/clips/0301_gate_euler25/writer_md5.txt 2>/dev/null | cut -d' ' -f1)
echo "GATE md5=$G want=72eb725755dd97457d9eebcebc7654a5 $( [ "$G" = 72eb725755dd97457d9eebcebc7654a5 ] && echo MATCH || echo MISMATCH)"
bash $D/run_driver_teacher_v1.sh $D/jobs_01_smoke.txt
echo "CHAIN_A_DONE $(date +%F_%T)"
