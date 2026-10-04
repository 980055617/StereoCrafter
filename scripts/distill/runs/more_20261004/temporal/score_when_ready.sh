#!/bin/bash
# Waits until every render a clip's final scoring needs exists, then runs score_clip_final.sh for it (in this order:
# 0170ctx, 0259, 0042, 0301).  Each scorer takes the GPU-1 lock per job.
set -u
cd /home/kawa/master_project/StereoCrafter
T=scripts/distill/runs/more_20261004/temporal
for C in 0170ctx 0259 0042 0301; do
  if [ "$C" = "0170ctx" ]; then L=$T/lpips_list_t3_0170_ctx.txt; S=$T/specs_t3_0170_ctx.txt; else L=$T/lpips_list_t2_$C.txt; S=$T/specs_t2_$C.txt; fi
  while :; do
    miss=0
    for p in $( (cat $L; cat $S) | grep -v '^#' | cut -d= -f2- | cut -d'#' -f1 | sort -u); do [ -f "$p" ] || miss=$((miss+1)); done
    [ $miss -eq 0 ] && break
    sleep 30
  done
  echo "READY $C $(date +%T) -> scoring"
  bash $T/score_clip_final.sh $C
  echo "SCORED $C $(date +%T)"
done
