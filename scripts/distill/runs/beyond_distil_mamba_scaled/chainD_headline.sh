#!/bin/bash
set -u
cd /home/kawa/master_project/StereoCrafter
D=scripts/distill/runs/beyond_distil_mamba_scaled
DELIV=$1
export SK_UNET=$DELIV
# (1) pixel-identity control: 0040 already has a step800 HOOK render (SK_UNET=shipped + SK_CK=step800).
#     Re-render it with the MERGED file through the tracked entry point; the pre-encode writer_md5
#     digests must match, which is what proves the shipped file IS the model the ladder measured.
J=$D/jobs_identity.txt; echo "0040 mstudent2_step800_deliv_ll none" > $J
bash $D/run_driver_v3.sh $J 1
# (2) the headline row
J=$D/jobs_headline.txt; : > $J
for C in 0042 0052 0125 0128 0141 0147 0170 0204 0225 0251 0259 0301; do
  echo "$C mstudent2_step800_deliv_ll none" >> $J
done
bash $D/run_driver_v3.sh $J 1
echo CHAIND_DONE $(date +%T)
