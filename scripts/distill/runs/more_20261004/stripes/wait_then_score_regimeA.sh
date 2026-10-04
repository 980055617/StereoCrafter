cd /home/kawa/master_project/StereoCrafter
until compgen -G "outputs/more_20261004/stripes/clips/0147_origin_g101_s8_A_rowlin/*_sbs.mkv" > /dev/null && [ -s outputs/more_20261004/stripes/clips/0147_origin_g101_s8_A_rowlin/speed_log.json ]; do sleep 10; done
TEMPORAL=1 scripts/distill/runs/more_20261004/stripes/score_subset_v1.sh regimeA outputs/more_20261004/stripes/specs_regimeA.txt
