#!/usr/bin/env python
"""Score a lane of clips for ALL SIX configs with the canonical scorer, unmodified.

  scorer: scripts/distill/runs/fulldata_v2/beyond4/score_clip_ll.py   SCORE_STEP=4
  path  : the LOSSLESS FFV1 renders only.  The four reused rows (origin / shipped Mamba /
          step800 deliverable / origin+s25) are re-scored rather than transcribed, so the
          published reference means (0.3933 / 0.3927 / 0.3804 / 0.3786) act as a harness anchor.

Specs are emitted clip-major because score_clip_ll.py caches the GT tile per clip.
usage: score_rungs.py <outfile> <clip> [clip ...]
"""
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import holelib as HL  # noqa: E402

out = sys.argv[1]
clips = sys.argv[2:]
specs = []
for c in clips:
    for lab, p in HL.panels(c):
        specs.append(f"{c}={p}")
print(f"[score] {len(specs)} specs over {len(clips)} clips -> {out}", flush=True)
cmd = [sys.executable, "scripts/distill/runs/fulldata_v2/beyond4/score_clip_ll.py"] + specs
env = dict(os.environ, SCORE_STEP="4")
with open(out, "w") as fh:
    rc = subprocess.call(cmd, cwd=HL.REPO, env=env, stdout=fh, stderr=subprocess.STDOUT)
print(f"[score] rc={rc}", flush=True)
sys.exit(rc)
