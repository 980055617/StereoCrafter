#!/bin/bash
# usage: mkjobs_fs_v1.sh <out jobfile> <model deliv|origin> <SCHED name from sched_defs.txt> <clip> [<clip> ...]
# writes lines "CLIP <model>_g100_<SCHED>pad <model> 1.00 <sigmas> 8"; refuses to overwrite
set -u
D=/home/kawa/master_project/StereoCrafter/scripts/distill/runs/more_20261004/fewer_steps
OUTF=$1; MODEL=$2; SCH=$3; shift 3
SIG=$(awk -v s=$SCH '$1==s{print $2}' $D/sched_defs.txt)
[ -n "$SIG" ] || { echo "unknown schedule $SCH"; exit 2; }
for C in "$@"; do echo "$C ${MODEL}_g100_${SCH}pad $MODEL 1.00 $SIG 8"; done >> $OUTF
