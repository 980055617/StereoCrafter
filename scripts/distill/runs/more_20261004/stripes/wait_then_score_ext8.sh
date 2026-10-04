# wait for the last deliverable-A extension render (0259), then LPIPS-score the 8 extension clips (with both baselines)
cd /home/kawa/master_project/StereoCrafter
until compgen -G "outputs/more_20261004/stripes/clips/0259_deliv_g100_T5nat_A_rowlin/*_sbs.mkv" > /dev/null && [ -s outputs/more_20261004/stripes/clips/0259_deliv_g100_T5nat_A_rowlin/speed_log.json ]; do sleep 10; done
scripts/distill/runs/more_20261004/stripes/score_lpips_v1.sh outputs/more_20261004/stripes/SCORES_ext8.txt outputs/more_20261004/stripes/lists_ext8_scorelist.txt
