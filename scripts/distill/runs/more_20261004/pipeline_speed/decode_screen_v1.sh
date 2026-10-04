#!/bin/bash
# Screen VAE-decode variants on saved latents; ONE lock acquisition for the whole screen (timing measurement).
# usage: decode_screen_v1.sh <latent_dir> <out_dir (new)> <windows e.g. 0,1,2,3> <T>
set -u
cd /home/kawa/master_project/StereoCrafter
PY=/home/kawa/miniconda3/envs/stereocrafter/bin/python
LAT=$1; OUTD=$2; WINS=$3; TT=$4
[ -e "$OUTD" ] && { echo "refusing to reuse $OUTD"; exit 3; }
mkdir -p $OUTD
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
flock /tmp/claude-gpu0.lock bash -c "
  echo LOCKED \$(date +%F_%T) load1=\$(cut -d' ' -f1 /proc/loadavg) >> $OUTD/screen.log
  for v in 'CHUNK=2 CL=0 BENCH=0 SKIP=0 DT=bf16' 'CHUNK=2 CL=0 BENCH=0 SKIP=1 DT=bf16' \
           'CHUNK=2 CL=1 BENCH=0 SKIP=0 DT=bf16' 'CHUNK=2 CL=0 BENCH=1 SKIP=0 DT=bf16' 'CHUNK=2 CL=1 BENCH=1 SKIP=0 DT=bf16' \
           'CHUNK=4 CL=0 BENCH=0 SKIP=0 DT=bf16' 'CHUNK=8 CL=0 BENCH=0 SKIP=0 DT=bf16' 'CHUNK=14 CL=0 BENCH=0 SKIP=0 DT=bf16' \
           'CHUNK=14 CL=1 BENCH=0 SKIP=0 DT=bf16' 'CHUNK=2 CL=0 BENCH=0 SKIP=0 DT=fp16' \
           'CHUNK=2 CL=0 BENCH=0 SKIP=0 DT=bf16'; do
    eval \$v
    TAG=c\${CHUNK}_cl\${CL}_b\${BENCH}_s\${SKIP}_\${DT}
    J=$OUTD/\$TAG.json; [ -e \$J ] && J=$OUTD/\${TAG}_rep2.json
    DB_LAT=$LAT DB_WINDOWS=$WINS DB_T=$TT DB_CHUNK=\$CHUNK DB_CL=\$CL DB_BENCH=\$BENCH DB_SKIP=\$SKIP DB_DTYPE=\$DT DB_OUT=\$J \
      $PY scripts/distill/runs/more_20261004/pipeline_speed/decode_bench_v1.py >> $OUTD/screen.log 2>&1
    echo \"rc=\$? \$TAG \$(date +%T) load1=\$(cut -d' ' -f1 /proc/loadavg)\" >> $OUTD/screen.log
  done
  echo UNLOCK \$(date +%F_%T) >> $OUTD/screen.log
"
