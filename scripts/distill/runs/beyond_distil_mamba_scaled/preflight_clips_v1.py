#!/usr/bin/env python
"""Pre-flight: every candidate TRAIN-split clip must decode through the DEPLOYED input path
(utils.inpainting.read_and_prepare_video + inpainting_inference._center_crop_frames) at 576x1024,
and must yield the expected dense 14-frame/stride-11 window list.  CPU only, no GPU, read-only.
usage: BD_CLIPS=a,b,c preflight_clips_v1.py
"""
import os, sys, json
REPO = "/home/kawa/master_project/StereoCrafter"
sys.path.insert(0, REPO); os.chdir(REPO)
sys.path.insert(0, os.path.join(REPO, "scripts/distill/runs/diag_trainer/mech"))
import mechlib as M

CLIPS = os.environ["BD_CLIPS"].split(",")
SPLIT = json.load(open("scripts/distill/splits/fulldata_v1.json"))
TEST, DEV = set(SPLIT["test"]), set(SPLIT["dev"])
tot = 0
for c in CLIPS:
    assert c not in TEST, f"{c} is a TEST clip -- refused"
    assert c not in DEV, f"{c} is a DEV clip -- refused"
    assert SPLIT["clips"][c]["role"] == "train", f"{c} role={SPLIT['clips'][c]['role']} -- refused"
    M.CLIP = c
    M.SPLAT = f"video_data/splatting/{c}_splatting_results.mp4"
    fps, left, cond, mask = M.read_deployed_inputs()
    w = M.window_starts(cond.shape[0])
    assert cond.shape[1:] == (3, 576, 1024), (c, cond.shape)
    assert mask.shape[1:] == (3, 576, 1024) or mask.shape[1] in (1, 3), (c, mask.shape)
    tot += len(w)
    print(f"OK {c} frames={cond.shape[0]:4d} cond={tuple(cond.shape)} mask={tuple(mask.shape)} "
          f"fps={fps:.3f} windows={len(w)} first/last={w[0]}/{w[-1]}", flush=True)
print(f"PREFLIGHT_OK {len(CLIPS)} clips, {tot} windows, est capture {tot*19.83/60:.1f} min")
