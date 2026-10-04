#!/bin/bash
# Per-clip final scoring (run once every render of that clip exists): LPIPS then temporal, each under the GPU-1 lock.
# usage: score_clip_final.sh 0259|0042|0301|0170ctx
set -u
cd /home/kawa/master_project/StereoCrafter
T=scripts/distill/runs/more_20261004/temporal
S=outputs/more_20261004/temporal/scores
C=$1
if [ "$C" = "0170ctx" ]; then
  bash $T/score_lpips_v1.sh $T/SCORES_LPIPS_T3_0170CTX.txt $T/lpips_list_t3_0170_ctx.txt
  bash $T/score_temporal_run.sh $S/t3_0170_ctx.json $T/specs_t3_0170_ctx.txt
else
  bash $T/score_lpips_v1.sh $T/SCORES_LPIPS_T2_$C.txt $T/lpips_list_t2_$C.txt
  bash $T/score_temporal_run.sh $S/t2_$C.json $T/specs_t2_$C.txt
fi
