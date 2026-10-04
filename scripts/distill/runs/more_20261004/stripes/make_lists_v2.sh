#!/bin/bash
# v2 = make_lists_v1.sh without the C (rowlin+shrink) labels (C == A by construction, PREREG_ADDENDUM_1.txt).
# Write the score / temporal / decomposition spec lists for the stripes lane (baselines first per clip, origin first).
# usage: make_lists_v2.sh <out_prefix> <clip> [<clip> ...]
#   -> <out_prefix>_scorelist.txt (clip=path), <out_prefix>_specs.txt (clip:label=path, for run_decomp_v1.py)
set -u
cd /home/kawa/master_project/StereoCrafter
PRE=$1; shift
SL=${PRE}_scorelist.txt; SP=${PRE}_specs.txt
for f in $SL $SP; do [ -e $f ] && { echo "refusing to overwrite $f"; exit 3; }; done
for c in "$@"; do
  for spec in "origin_ll=outputs/beyond4_lossless/clips/${c}_origin_ll" \
              "deliv_g100_T5nat=outputs/finalcheck_20261004/speed/clips/${c}_deliv_g100_T5nat" \
              "origin_g101_s8_A_rowlin=outputs/more_20261004/stripes/clips/${c}_origin_g101_s8_A_rowlin" \
              "origin_g101_s8_B_telea=outputs/more_20261004/stripes/clips/${c}_origin_g101_s8_B_telea" \
              "deliv_g100_T5nat_A_rowlin=outputs/more_20261004/stripes/clips/${c}_deliv_g100_T5nat_A_rowlin" \
              "deliv_g100_T5nat_B_telea=outputs/more_20261004/stripes/clips/${c}_deliv_g100_T5nat_B_telea"; do
    lab=${spec%%=*}; d=${spec#*=}
    p=$d/${c}_inpainting_results_sbs.mkv
    [ -f $p ] || { echo "MISSING $p"; exit 4; }
    echo "$c=$p" >> $SL
    echo "$c:$lab=$p" >> $SP
  done
done
wc -l $SL $SP
