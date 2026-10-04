#!/bin/bash
# Runs ONE render while the caller holds the GPU-0 flock; times the python process (start -> exit) inside the lock.
# args: PY RUN OD LOG DESC
set -u
PY=$1; RUN=$2; OD=$3; LOG=$4; DESC=$5
LOAD=$(cut -d' ' -f1 /proc/loadavg)
GPUS=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used,temperature.gpu,clocks.sm --format=csv,noheader | tr '\n' ';' | tr -d ' ')
T0=$(date +%s.%N)
/usr/bin/time -v $PY $RUN > $OD.log 2>&1
RC=$?
T1=$(date +%s.%N)
LOAD2=$(cut -d' ' -f1 /proc/loadavg)
SECS=$(awk -v a=$T0 -v b=$T1 'BEGIN{printf "%.2f", b-a}')
RSS=$(grep -oE 'Maximum resident set size \(kbytes\): [0-9]+' $OD.log | grep -oE '[0-9]+$')
MD5=$(cut -d' ' -f1 $OD/writer_md5.txt 2>/dev/null | head -1)
ST=$(grep -oE '^\[stage\] rep=0 run_s=[0-9.]+' $OD.log | head -1)
echo "RUN $DESC rc=$RC process_s=$SECS md5=$MD5 maxrss_kb=${RSS:-NA} load1=$LOAD->$LOAD2 gpus=$GPUS $(date +%F_%T) dir=$OD :: $ST" | tee -a $LOG
