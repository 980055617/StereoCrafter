#!/bin/bash
# CONTROL B extras (run after chain_regB_v1.sh):
#  (a) sample the regB best checkpoint with 25 Euler steps on 0301 (does the damage sit in the denoiser or in the deployed 8-step trajectory?)
#  (b) per-frame-registered training run (regpos_pf) + its render/score chain (closes the "approximate global registration" objection).
# usage: chain_regB_v3_extras.sh <best step of the regB run>      GPU 1, FFV1 writer, every render -> NEW dir.
set -u; cd /home/kawa/master_project/StereoCrafter
set +u; source ~/miniconda3/etc/profile.d/conda.sh; conda activate stereocrafter; set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=1 SCORE_STEP=4 LOSSLESS_SBS=1 KEEP_ANAGLYPH=0
B=scripts/distill/runs/clean_controls/regB; I=$B/infer; S=$B/scores; T=$B/train; BEST=$1
SCORER=scripts/distill/runs/fulldata_v2/beyond4/score_clip_ll.py
newdir() { local O=$1; local k=1; while [ -e "$O" ]; do k=$((k+1)); O=$1_$k; done; echo $O; }
# (a) 25-step sampling of the regB best checkpoint on 0301
O=$(newdir $I/regB_step${BEST}_0301_s25); echo "START regB_step${BEST}_0301_s25 (25 steps) -> $O $(date +%T)"
MINIFT_STEPS=25 MINIFT_CK=$T/regpos/step$BEST.pt python $I/xcheck_hybrid_regB_ll_steps.py e1all 0301 $O > $O.log 2>&1; RC=$?; echo "EXIT regB_step${BEST}_0301_s25 rc=$RC $(date +%T)"
if [ $RC -ne 0 ]; then echo "RETRY s25 (once)"; O=$(newdir $I/regB_step${BEST}_0301_s25); MINIFT_STEPS=25 MINIFT_CK=$T/regpos/step$BEST.pt python $I/xcheck_hybrid_regB_ll_steps.py e1all 0301 $O > $O.log 2>&1; echo "EXIT-RETRY rc=$? $(date +%T)"; fi
S25OUT=$O
echo "START score s25 $(date +%T)"
python $SCORER 0301=outputs/beyond4_lossless/clips/0301_origin_ll/0301_inpainting_results_sbs.mkv 0301=outputs/beyond4_lossless/clips/0301_s25_ll/0301_inpainting_results_sbs.mkv 0301=$I/regB_step${BEST}_0301/0301_inpainting_results_sbs.mkv 0301=$S25OUT/0301_inpainting_results_sbs.mkv 2>&1 | grep -viE "warning|setting up|loading model|self.load_state|^/home/kawa" | tee $S/scores_regB_s25.txt
# (b) per-frame-registered training + follow-up chain
echo "START train regpos_pf $(date +%T)"
python $T/xcheck_mini_ft_regB_pf.py regpos regpos_pf > $T/train_regpos_pf.log 2>&1; echo "EXIT train regpos_pf rc=$? $(date +%T)"
grep -q MINIFT_DONE $T/train_regpos_pf.log || { echo "TRAIN_PF_FAILED"; echo "CHAIN_EXTRAS_DONE $(date +%T)"; exit 1; }
PFDIR=$(grep '^OUT ' $T/train_regpos_pf.log | awk '{print $2}'); echo "PFDIR $PFDIR"
$I/chain_regB_v2_followup.sh $PFDIR regBpf
echo "CHAIN_EXTRAS_DONE $(date +%T)"
