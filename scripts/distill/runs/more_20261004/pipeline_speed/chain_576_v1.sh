#!/bin/bash
# 576x1024 chain: A1/A2 baselines -> decode screen on A1 latents -> B1 warm-up diagnosis -> B2 cuDNN/channels_last screens.
# Each render takes the GPU-0 lock per job (inside the driver); the decode screen holds it for the whole screen.
set -u
cd /home/kawa/master_project/StereoCrafter
L=scripts/distill/runs/more_20261004/pipeline_speed
O=outputs/more_20261004/pipeline_speed
bash $L/run_driver_stage_v2.sh $L/jobs_A_576.txt $O/a_576
[ -d $O/a_576/clips/0301_A1_deliv_T5_576/lat ] && bash $L/decode_screen_v1.sh $O/a_576/clips/0301_A1_deliv_T5_576/lat $O/d_screen_576 0,1,2,13 151
bash $L/run_driver_stage_v2.sh $L/jobs_B1_warmup.txt $O/b1_warmup
bash $L/run_driver_stage_v2.sh $L/jobs_B2_screen.txt $O/b2_screen
echo CHAIN_576_DONE $(date +%F_%T)
