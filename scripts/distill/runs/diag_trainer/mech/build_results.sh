#!/bin/bash
# Collate every measurement of this mech/ session into one plain-text artifact.
set -u; cd /home/kawa/master_project/StereoCrafter
H=scripts/distill/runs/diag_trainer/mech
set +u; source ~/miniconda3/etc/profile.d/conda.sh; conda activate stereocrafter; set -u
O=$H/RESULTS.txt; k=1; while [ -e "$O" ]; do k=$((k+1)); O=$H/RESULTS_$k.txt; done
{
echo "================================================================ TEST B: per-sampler-step x0-hat"
cat $H/testb/testb_table.txt
echo; echo "---------------- derived: where the sampler adds detail"
cat $H/testb/testb_refinement.txt
echo; echo "---------------- derived: exact Euler information weights of the deployed 8-step Karras grid"
cat $H/testb/euler_information_weights.txt
echo; echo "================================================================ P1 target decomposition (explains the sigma bisect)"
cat $H/derived/p1_target_decomposition.txt
echo; echo "================================================================ TEST A: on-trajectory runs"
python $H/summarize_mech.py $H/a1 $H/a2 $H/a2x0 $H/a2_hi5 2>&1
echo; echo "---------------- checkpoints vs origin"
python $H/check_ck_vs_origin.py $H/a1 $H/a2 $H/a2x0 $H/a2_hi5 2>&1
echo; echo "---------------- weight-delta magnitude and direction (corrected for the bf16 starting point)"
python $H/analyze_deltas_bf16base.py $H/a2 $H/a2x0 $H/a2_hi5 2>&1
echo; echo "---------------- (for reference) the OLD fp32-baseline numbers analyze_deltas.py prints"
python scripts/distill/runs/diag_trainer/minift/analyze_deltas.py $H/a2 $H/a2x0 $H/a2_hi5 2>&1
echo; echo "================================================================ sampled LPIPS (score_clip.py, whole clip 0301)"
cat $H/scores_mech.txt
} > $O 2>&1
echo "WROTE $O"
