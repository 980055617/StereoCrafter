"""blur_diag (deep_20261004) -- shared row loader.  Definitions: PREREG.txt (same dir).  Read-only on every input.

Every row is returned as uint8 [38, 576, 1024, 3] at the published scored frames (SCORE_STEP=4 grid, frames 0..148),
in the order of meta["frames"].  Render rows are the RIGHT half of a 576x2048 SBS FFV1 file; frame k of a render is
frame k of the GT (score_clip_ll.py convention).
"""
import json
import os

import cv2
import numpy as np
from decord import VideoReader, cpu

REPO = "/home/kawa/master_project/StereoCrafter"
os.chdir(REPO)
CACHE = "/mnt/ssd_data/deep_20261004/blur_diag/cache_r2"
OUT = "outputs/deep_20261004/blur_diag"
ER = "outputs/more_20261004/eval_robustness"
TH, TW = 576, 1024
DIL = 8
CLIPS = "0042 0052 0125 0128 0141 0147 0170 0204 0225 0251 0259 0301".split()
FMT4400 = CLIPS[:6]
FMT2160 = CLIPS[6:]

SBS_ROWS = {  # row -> eval_robustness config label (path read from its JSON)
    "ORIGIN": "origin_ll", "DELIV": "mstudent2_step800_deliv_ll", "S25": "s25_ll",
    "T5NAT": "deliv_g100_T5nat", "T5PAD": "deliv_g100_T5pad",
}
VAE_ROWS = ["VAE_GT", "VAE_GT32", "VAE_GTx", "RS_GTx", "VAE_BR"]
GT_GEOMETRY = {"GT", "VAE_GT", "VAE_GT32", "VAE_GTx", "RS_GTx", "RS_GTx_L"}       # everything else lives in render geometry


def er_json(clip):
    p = f"{ER}/score_v1_wide/{clip}.json" if clip == "0125" else f"{ER}/score_v1/{clip}.json"
    return p, json.load(open(p))


def meta(clip):
    return json.load(open(f"{CACHE}/{clip}/meta.json"))


def hires_a_path(clip):
    return f"{OUT}/hiresA_crop_r2/{clip}_origin_hiresA_1024x1792/{clip}_inpainting_results_sbs.mkv"


def hires_b_path(clip):
    return f"{OUT}/hiresB_r2/clips/{clip}_origin_upx175/{clip}_inpainting_results_sbs.mkv"


def hires_b_l_path(clip):   # PREREG ADDENDUM 2 (R4-L): LANCZOS-downsampled HIRES_B
    return f"{OUT}/hiresB_r2/lanczos/{clip}_origin_upx175_L/{clip}_inpainting_results_sbs.mkv"


def rs_gtx_l_path(clip):    # PREREG ADDENDUM 2 (R4-L): resampling control with LANCZOS down
    return f"{OUT}/lanczos_r2/{clip}/{clip}_RS_GTx_L.mkv"


def row_paths(clip):
    """row -> (kind, path) for every row that exists for this clip."""
    _, J = er_json(clip)
    out = {}
    for r, lab in SBS_ROWS.items():
        out[r] = ("sbs", J["configs"][lab]["path"])
    for r in VAE_ROWS:
        p = f"{OUT}/vae_rt_r2/{clip}/{clip}_{r}.mkv"
        if os.path.exists(p):
            out[r] = ("single", p)
    for r, p in (("HIRES_A", hires_a_path(clip)), ("HIRES_B", hires_b_path(clip)), ("HIRES_B_L", hires_b_l_path(clip))):
        if os.path.exists(p):
            out[r] = ("sbs", p)
    if os.path.exists(rs_gtx_l_path(clip)):
        out["RS_GTx_L"] = ("single", rs_gtx_l_path(clip))
    return out


def read_frames(path, frames):
    vr = VideoReader(path, ctx=cpu(0))
    assert max(frames) < len(vr), (path, len(vr))
    return vr.get_batch(list(frames)).asnumpy()


def load_row(clip, row, frames, paths=None):
    """uint8 [n,576,1024,3]."""
    if row == "GT":
        return np.ascontiguousarray(np.load(f"{CACHE}/{clip}/GTreg.npy", mmap_mode="r")[frames])
    if row == "BR":
        return np.ascontiguousarray(np.load(f"{CACHE}/{clip}/BR.npy", mmap_mode="r")[frames])
    paths = paths or row_paths(clip)
    if row == "LEFT":
        a = read_frames(paths["ORIGIN"][1], frames)
        assert a.shape[1:] == (TH, 2 * TW, 3), a.shape
        return np.ascontiguousarray(a[:, :, :TW])
    kind, p = paths[row]
    a = read_frames(p, frames)
    if kind == "sbs":
        assert a.shape[1:] == (TH, 2 * TW, 3), (row, a.shape)
        return np.ascontiguousarray(a[:, :, TW:])
    assert a.shape[1:] == (TH, TW, 3), (row, a.shape)
    return a


def load_left(clip, row, frames, paths=None):
    paths = paths or row_paths(clip)
    kind, p = paths[row]
    assert kind == "sbs"
    return np.ascontiguousarray(read_frames(p, frames)[:, :, :TW])


def holes(clip, frames):
    """(hole bool, dilated-hole bool) [n,576,1024] at the scored frames."""
    h = np.load(f"{CACHE}/{clip}/HOLE.npy", mmap_mode="r")[frames]
    k = np.ones((2 * DIL + 1, 2 * DIL + 1), np.uint8)
    d = np.stack([cv2.dilate(x.astype(np.uint8), k) > 0 for x in h])
    return np.ascontiguousarray(h), d


def composite(x, ref, dil):
    """Diagnostic composite: dilated-hole pixels of x replaced by ref (evaluation only, never an output)."""
    return np.where(dil[..., None], ref, x)
