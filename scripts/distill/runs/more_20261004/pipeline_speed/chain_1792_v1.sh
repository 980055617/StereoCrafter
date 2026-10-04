#!/bin/bash
# 1024x1792 chain: A3/A4 baselines -> A3 autotune snapshot -> decode screen on A3 latents -> encode batch test (576 and
# 1792) -> C2 identity-fix render.  Renders take the GPU-0 lock per job; each screen/test holds it for its duration.
set -u
cd /home/kawa/master_project/StereoCrafter
L=scripts/distill/runs/more_20261004/pipeline_speed
O=outputs/more_20261004/pipeline_speed
PY=/home/kawa/miniconda3/envs/stereocrafter/bin/python
bash $L/run_driver_stage_v2.sh $L/jobs_A_1792.txt $O/a_1792
$PY - <<'PYEOF'
import json, os
S = json.load(open("outputs/more_20261004/pipeline_speed/a_1792/clips/0301_A3_deliv_T5_1792/stage_log.json"))
out = "outputs/more_20261004/pipeline_speed/autotune_choices_A3_1792.json"
assert not os.path.exists(out)
json.dump(S["autotune"]["snapshot"], open(out, "w"), indent=1, sort_keys=True)
print("wrote", out, {k.split(".")[-1]: [e["idx"] for e in v] for k, v in S["autotune"]["snapshot"].items()})
PYEOF
[ -d $O/a_1792/clips/0301_A3_deliv_T5_1792/lat ] && bash $L/decode_screen_v2.sh $O/a_1792/clips/0301_A3_deliv_T5_1792/lat $O/d_screen_1792 1,13 151 \
  "2 0 0 0 bf16;2 0 0 1 bf16;2 0 1 0 bf16;2 1 0 0 bf16;4 0 0 0 bf16;14 0 0 0 bf16;2 0 0 0 bf16"
mkdir -p $O/e_encbatch
CUDA_VISIBLE_DEVICES=0 flock /tmp/claude-gpu0.lock $PY $L/enc_batch_test_v1.py 576 1024 $O/e_encbatch/encbatch_576.json > $O/e_encbatch/encbatch_576.log 2>&1
CUDA_VISIBLE_DEVICES=0 flock /tmp/claude-gpu0.lock $PY $L/enc_batch_test_v1.py 1024 1792 $O/e_encbatch/encbatch_1792.json > $O/e_encbatch/encbatch_1792.log 2>&1
bash $L/run_driver_stage_v2.sh $L/jobs_C_1792.txt $O/c_1792
echo CHAIN_1792_DONE $(date +%F_%T)
