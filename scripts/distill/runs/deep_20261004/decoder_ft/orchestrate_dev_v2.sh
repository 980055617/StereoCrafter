#!/bin/bash
# decoder_ft: DEV orchestration v2 (v1 stopped at 11:33 while only waiting: its GPU0 re-decode had failed at checkpoint
# load -- weights_only=True rejected the TorchVersion in the checkpoint config; redecode/roundtrip now load this lane's own
# checkpoints with weights_only=False).  Same stages, same order.
# decoder_ft: DEV orchestration of chain_dev_ckpt_v1.sh stages over both GPUs (each stage takes its GPU's lock per call).
#   GPU0: after the dev stock chain and step1000.pt exist -> re-decode s250 s500 s1000
#   GPU1: after TRAIN_DONE -> re-decode s1500 s2000 -> C3 step-0 control -> score_clip_ll on the 12 stock rows
#   then registered + aux scorers split by clip over both GPUs -> pre-registered selection (analyze)
#   then the decoder-only dev GT round trip (diagnostic only, after the selection is written)
set -u
cd /home/kawa/master_project/StereoCrafter
L=scripts/distill/runs/deep_20261004/decoder_ft
RUN=main_v1
CK=/mnt/ssd_data/deep_20261004/decoder_ft/ck/$RUN
STEPS="250 500 1000 1500 2000"
C=$L/chain_dev_ckpt_v1.sh
wait_for() { until grep -q "$2" "$1" 2>/dev/null; do sleep 20; done; }
echo "ORCH_START $(date +%F_%T)"
wait_for $L/chain_dev_stock_v1.log DEV_STOCK_DONE
echo "dev stock chain done $(date +%F_%T)"
( until [ -e $CK/step1000.pt ]; do sleep 20; done; bash $C redec 0 $RUN "250 500 1000" ) > $L/orch_dev_v2_gpu0.log 2>&1 &
PA=$!
( wait_for $L/train_main_v1.log TRAIN_DONE; bash $C redec 1 $RUN "1500 2000"; bash $C c3 1 $RUN;
  bash $C score 1 $RUN "$STEPS" "0040 0082 0091 0184 0245 0268" ) > $L/orch_dev_v2_gpu1.log 2>&1 &
PB=$!
wait $PA $PB
echo "redecode + stock UNREG done $(date +%F_%T)"
( bash $C reg 0 $RUN "$STEPS" "0040 0082 0091" ) > $L/orch_dev_v2_reg0.log 2>&1 &
PC=$!
( bash $C reg 1 $RUN "$STEPS" "0184 0245 0268" ) > $L/orch_dev_v2_reg1.log 2>&1 &
PD=$!
wait $PC $PD
echo "registered + aux done $(date +%F_%T)"
bash $C analyze $RUN "$STEPS"
SPEC=""; for s in $STEPS; do SPEC="$SPEC,s$s=$CK/step$s.pt"; done; SPEC=${SPEC#,}
set +u; source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate stereocrafter; set -u
CUDA_VISIBLE_DEVICES=1 flock /tmp/claude-gpu1.lock env PYTHONPATH=/mnt/ssd_data/deep_20261004/skeptic/pylib \
  TORCH_HOME=/mnt/ssd_data/deep_20261004/decoder_ft/torch_home python $L/roundtrip_v1.py $L/ROUNDTRIP_DEV_${RUN}.json dev \
  0040,0082,0091,0184,0245,0268 $SPEC > $L/roundtrip_dev_${RUN}.log 2>&1 < /dev/null
echo "ORCH_DEV_DONE roundtrip rc=$? $(date +%F_%T)"
