#!/bin/bash
# emit score-list lines "CLIP INPUT origin=<sbs> deliv=<sbs>" for the given clip and inputs (renders <clip>_<INPUT>_<model>)
C=outputs/deep_20261004/input_side/clips
clip=$1; shift
for inp in "$@"; do echo "$clip $inp origin=$C/${clip}_${inp}_origin/${clip}_inpainting_results_sbs.mkv deliv=$C/${clip}_${inp}_deliv/${clip}_inpainting_results_sbs.mkv"; done
