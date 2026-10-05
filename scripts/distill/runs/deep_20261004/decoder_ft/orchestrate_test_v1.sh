#!/bin/bash
# decoder_ft: TEST orchestration (ONE evaluation) of chain_test_v1.sh stages over both GPUs.
# usage: bash orchestrate_test_v1.sh <run> <sel> <us>     (sel / us = the dev selections in SELECTION_DEV_<run>.json)
set -u
cd /home/kawa/master_project/StereoCrafter
L=scripts/distill/runs/deep_20261004/decoder_ft
RUN=$1; SEL=$2; US=$3
C=$L/chain_test_v1.sh
A6="0042 0052 0125 0128 0141 0147"
B6="0170 0204 0225 0251 0259 0301"
echo "ORCH_TEST_START run=$RUN sel=$SEL us=$US $(date +%F_%T)"
( bash $C redec 0 $RUN $SEL "$A6" ) > $L/orch_test_redec0.log 2>&1 &
P1=$!
( bash $C redec 1 $RUN $SEL "$B6" ) > $L/orch_test_redec1.log 2>&1 &
P2=$!
wait $P1 $P2
bash $C gates
bash $C unsharp $US "$A6 $B6"
( bash $C unreg 0 $RUN $SEL $US "$A6" ) > $L/orch_test_unreg0.log 2>&1 &
P1=$!
( bash $C unreg 1 $RUN $SEL $US "$B6" ) > $L/orch_test_unreg1.log 2>&1 &
P2=$!
wait $P1 $P2
bash $C rows $RUN $SEL $US || { echo "ROWS FAILED"; exit 1; }
( bash $C score 0 $RUN $SEL "$A6" ) > $L/orch_test_score0.log 2>&1 &
P1=$!
( bash $C score 1 $RUN $SEL "$B6" ) > $L/orch_test_score1.log 2>&1 &
P2=$!
wait $P1 $P2
bash $C roundtrip 1 $RUN $SEL
bash $C panels $RUN $SEL
echo "ORCH_TEST_DONE $(date +%F_%T)"
