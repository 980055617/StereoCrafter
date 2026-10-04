#!/bin/bash
# [v2 = decode_screen_v1.sh with the variant list as argument 5 (';'-separated "CHUNK CL BENCH SKIP DT" tuples)]
# Screen VAE-decode variants on saved latents; ONE lock acquisition for the whole screen (timing measurement).
# usage: decode_screen_v2.sh <latent_dir> <out_dir (new)> <windows> <T> "<c cl b s dt>;<c cl b s dt>;..."
set -u
cd /home/kawa/master_project/StereoCrafter
PY=/home/kawa/miniconda3/envs/stereocrafter/bin/python
LAT=$1; OUTD=$2; WINS=$3; TT=$4; VARS=$5
[ -e "$OUTD" ] && { echo "refusing to reuse $OUTD"; exit 3; }
mkdir -p $OUTD
echo "$VARS" > $OUTD/variants.txt
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
flock /tmp/claude-gpu0.lock bash -c "
  echo LOCKED \$(date +%F_%T) load1=\$(cut -d' ' -f1 /proc/loadavg) >> $OUTD/screen.log
  IFS=';' read -ra VV <<< \"$VARS\"
  for v in \"\${VV[@]}\"; do
    read -r CHUNK CL BENCH SKIP DT <<< \"\$v\"
    TAG=c\${CHUNK}_cl\${CL}_b\${BENCH}_s\${SKIP}_\${DT}
    J=$OUTD/\$TAG.json; [ -e \$J ] && J=$OUTD/\${TAG}_rep2.json
    DB_LAT=$LAT DB_WINDOWS=$WINS DB_T=$TT DB_CHUNK=\$CHUNK DB_CL=\$CL DB_BENCH=\$BENCH DB_SKIP=\$SKIP DB_DTYPE=\$DT DB_OUT=\$J \
      $PY scripts/distill/runs/more_20261004/pipeline_speed/decode_bench_v1.py >> $OUTD/screen.log 2>&1
    echo \"rc=\$? \$TAG \$(date +%T) load1=\$(cut -d' ' -f1 /proc/loadavg)\" >> $OUTD/screen.log
  done
  echo UNLOCK \$(date +%F_%T) >> $OUTD/screen.log
"
