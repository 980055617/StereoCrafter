#!/bin/bash
# after the v2 GT fine-tune: evaluate e006 / e004 / e002 on real GT (12 test clips + 0160) against origin and the v2 seed
set -u; cd /home/kawa/master_project/StereoCrafter; D=scripts/distill; R=$D/runs/fulldata_v2
until ! systemctl --user is-active --quiet 'gt_ft_v2_1*'; do sleep 300; done
W=$(ls -d /mnt/ssd_data/stereocrafter_weights/GTfinetune_v2_light28/MambaCrafter_*/ | tail -1); echo "run dir $W"; ls $W | grep -E 'train_state_epoch' | tr '\n' ' '; echo
[ -n "$W" ] && [ -d "$W" ] || { echo "no run dir found; refusing to continue (the cleanup at the end would run in the repo root)"; exit 1; }
RK=$(ls -t logs/*_rank0.log | head -1); grep -oE 'Epoch [0-9]+/6 done \| avg_loss=[0-9.]+' "$RK" | tr '\n' ' '; echo
for E in 006 004 002; do [ -f ${W}train_state_epoch000$E.pt ] || continue; $D/eval_v2.sh ${W}train_state_epoch000$E.pt gt_v2_e$E 2>&1 | grep -E 'REALGT_SUMMARY|WARN|Traceback'; done
rm -rf ${W}deepspeed_state_epoch0000* 2>/dev/null; du -sh $W
echo "GT_FT_V2_EVAL_DONE $(date +%H:%M:%S)"
