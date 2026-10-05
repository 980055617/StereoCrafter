#!/bin/bash
# vae_20261005 / decoder_swap -- DEV orchestration (PREREG.txt sections 2-3), one role per GPU, run as transient systemd units.
#   gpu0: redec 0082 0091 0184 0245 (all decoders) -> [wait gpu1 redec] gate D1 -> D3 -> unreg 0040 0082 0091
#         -> [wait gpu1 unreg] rows -> reg+aux 0040 0082 0091 -> temporal 0040 0082 0091 -> DS-SEED 0040 deliv
#   gpu1: headroom 0082 0091 0184 0245 0268 (+ HR-SEED 0268) -> redec 0268 (all decoders) -> unreg 0184 0245 0268
#         -> [wait rows] reg+aux 0184 0245 0268 -> temporal 0184 0245 0268 -> DS-SEED 0268 deliv -> headroom scores (6 clips)
# Every GPU call holds its GPU's lock (run_gpu_v1.sh).  Synchronisation by marker files in $L/sync_dev_v1/.
set -u
ROLE=$1
cd /home/kawa/master_project/StereoCrafter
L=scripts/distill/runs/vae_20261005/decoder_swap
C=$L/chain_dev_v1.sh
S=$L/sync_dev_v1
O=outputs/vae_20261005/decoder_swap
mkdir -p $S
DECS="stock stock32 ftmse ftema cd"
waitf() { until [ -e "$1" ]; do sleep 20; done; }
echo "ORCH $ROLE START $(date +%F_%T)"
case $ROLE in
  gpu0)
    bash $C redec 0 "$DECS" "0082 0091 0184 0245"
    touch $S/redec_gpu0.done
    waitf $S/redec_gpu1.done
    bash $C gate
    bash $L/run_gpu_v1.sh 0 $L/cd_determinism_v1.log $L/cd_determinism_v1.py $L/GATE_D3_cd_determinism.txt 0040 deliv_cap
    echo "D3 rc=$? $(date +%F_%T)"
    bash $C unreg 0 "0040 0082 0091"
    touch $S/unreg_gpu0.done
    waitf $S/unreg_gpu1.done
    bash $C rows
    touch $S/rows.done
    bash $C reg 0 "0040 0082 0091"
    bash $C temporal 0 "0040 0082 0091"
    bash $L/run_gpu_v1.sh 0 $L/seed_diag_v1.log $L/seed_diag_v1.py $O/seed_diag_v1 0040 deliv_cap
    echo "DS-SEED 0040 rc=$? $(date +%F_%T)"
    touch $S/gpu0.done ;;
  gpu1)
    bash $L/run_gpu_v1.sh 1 $L/headroom_dev_v1.log $L/headroom_v1.py stock,stock32,ftmse,ftema,sd15,cd 0082,0091,0184,0245,0268
    echo "HEADROOM rc=$? $(date +%F_%T)"
    bash $L/run_gpu_v1.sh 1 $L/headroom_dev_v1.log $L/headroom_v1.py cd@1,cd@2 0268
    echo "HEADROOM SEED rc=$? $(date +%F_%T)"
    bash $C redec 1 "$DECS" "0268"
    touch $S/redec_gpu1.done
    bash $C unreg 1 "0184 0245 0268"
    touch $S/unreg_gpu1.done
    waitf $S/rows.done
    bash $C reg 1 "0184 0245 0268"
    bash $C temporal 1 "0184 0245 0268"
    bash $L/run_gpu_v1.sh 1 $L/seed_diag_v1.log $L/seed_diag_v1.py $O/seed_diag_v1 0268 deliv_cap
    echo "DS-SEED 0268 rc=$? $(date +%F_%T)"
    mkdir -p $O/headroom_dev_v1
    for c in 0040 0082 0091 0184 0245 0268; do
      [ -e $O/headroom_dev_v1/$c.json ] && { echo "headroom score $c exists (smoke), kept"; continue; }
      bash $L/run_gpu_v1.sh 1 $L/score_headroom_v1.log $L/score_headroom_v1.py $O/headroom_dev_v1 $c
      echo "HEADROOM SCORE $c rc=$? $(date +%F_%T)"
    done
    touch $S/gpu1.done ;;
  *) echo "unknown role"; exit 1 ;;
esac
echo "ORCH $ROLE DONE $(date +%F_%T)"
