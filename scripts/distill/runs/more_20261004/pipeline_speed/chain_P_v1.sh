#!/bin/bash
# F10 (torch.compile of the VAE decoder) renders + the LPIPS gate.  Waits for the C0 identity pre-check first.
set -u
cd /home/kawa/master_project/StereoCrafter
L=scripts/distill/runs/more_20261004/pipeline_speed
O=outputs/more_20261004/pipeline_speed
until [ -s $O/c0_576/clips/0301_C0_deliv_T5_576_idfix_2win/writer_md5.txt ] || grep -q "rc=[1-9]" $O/c0_576/timing_gpu0.txt 2>/dev/null; do sleep 10; done
M=$(cut -d' ' -f1 $O/c0_576/clips/0301_C0_deliv_T5_576_idfix_2win/writer_md5.txt 2>/dev/null | head -1)
[ "$M" = "719d6ceb1d9c0d083e4e43754a4ec3e9" ] || { echo "C0 pre-check failed ($M) -- P renders not run"; exit 2; }
echo "C0 ok, starting P renders $(date +%T)"
bash $L/run_driver_stage_v3.sh $L/jobs_P_compile.txt $O/p_576
P1=$O/p_576/clips/0301_P1_deliv_T5_576_idfix_vaecompile_cold/0301_inpainting_results_sbs.mkv
if [ -f $P1 ]; then
  printf "0301=outputs/finalcheck_20261004/speed/clips/0301_deliv_g100_T5nat/0301_inpainting_results_sbs.mkv\n0301=%s\n" $P1 > $O/p_576/scorelist_P1.txt
  bash $L/score_v1.sh $O/p_576/SCORES_P1_vs_ref.txt $O/p_576/scorelist_P1.txt
fi
until [ -s $O/autotune_choices_A3_1792.json ]; do sleep 20; done
bash $L/run_driver_stage_v3.sh $L/jobs_P3_compile_1792.txt $O/p_1792
echo CHAIN_P_DONE $(date +%F_%T)
