"""CONTROL A scoring hook = scripts/distill/runs/diag_trainer/minift/xcheck_hybrid_minift.py (MINIFT_CK tensor swap at sampling
time; modes originall / e1all / e1high / e1low / e1range:lo:hi) + scripts/distill/runs/fulldata_v2/beyond4/infer_lossless.py's
writer rebinding (the _sbs output is written as FFV1 level 3 / bgr0 / -g 1 Matroska, pre-encode md5 appended to
<save_dir>/writer_md5.txt and written to <file>.mkv.md5; anaglyph skipped unless KEEP_ANAGLYPH=1).
Nothing in the tracked repo is modified: `inpainting_inference.write_video_opencv` is rebound in that module's namespace and
`_Pipe.__call__` is wrapped exactly as in the minift hook, so the sampling path is the shipped one
(config/0160_overfit_inference_matched.json: 8 steps, guidance 1.01, 14-frame windows overlap 3, noise_seed 1234).
   mode originall : the swap-set is restored to origin at every UNet call (control: must reproduce the lossless origin md5)
   mode e1all     : MINIFT_CK tensors at every sampler step
   mode e1high    : MINIFT_CK tensors only when t = 0.25 ln sigma > 1.0 (sigma 700 / 286.5 / 102.9 = first 3 of 8 steps)
   mode e1low     : MINIFT_CK tensors only when t <= 1.0 (sigma 31.0 / 7.28 / 1.17 / 0.097 / 0.002 = last 5 steps)
usage: MINIFT_CK=<state-dict .pt> [LOSSLESS_SBS=1] python hybrid_ll.py <mode> <clip> <save_dir>
"""
import os, sys, hashlib, subprocess, collections
import numpy as np
ROOT = "/home/kawa/master_project/StereoCrafter"; sys.path.insert(0, ROOT); os.chdir(ROOT)
os.environ.setdefault("MAMBA_SELF_ATTN_INCLUDE", "__nomatch__")
import torch
import inpainting_inference as ii
from utils.inpainting import write_video_opencv as _orig_write

FFMPEG = os.environ.get("FFMPEG_BIN", "/home/kawa/miniconda3/envs/stereocrafter/bin/ffmpeg")
LOSSLESS_SBS = os.environ.get("LOSSLESS_SBS", "1") not in ("0", "", "false", "False")
KEEP_ANAGLYPH = os.environ.get("KEEP_ANAGLYPH", "0") not in ("0", "", "false", "False")

# ---------------- FFV1 writer, verbatim from beyond4/infer_lossless.py ----------------
def _ffv1_write(arr: np.ndarray, fps: float, path: str) -> None:
    if arr.dtype != np.uint8:
        raise TypeError(f"lossless writer expects uint8, got {arr.dtype}")
    T, H, W, C = arr.shape
    if C != 3:
        raise ValueError(f"expected 3 channels, got {C}")
    cmd = [FFMPEG, "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", f"{float(fps):.6f}", "-i", "-",
           "-an", "-c:v", "ffv1", "-level", "3", "-g", "1", "-slicecrc", "1", "-threads", "8",
           "-pix_fmt", "bgr0", path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(T):
        proc.stdin.write(np.ascontiguousarray(arr[i]).tobytes())
    proc.stdin.close()
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"ffmpeg FFV1 encode failed rc={rc}: {' '.join(cmd)}")

def _patched_write(input_frames, fps, output_video_path):
    arr = np.ascontiguousarray(input_frames)
    base = os.path.basename(output_video_path); save_dir = os.path.dirname(output_video_path)
    digest = hashlib.md5(arr.tobytes()).hexdigest()
    with open(os.path.join(save_dir, "writer_md5.txt"), "a") as fh:
        fh.write(f"{digest} {tuple(arr.shape)} {arr.dtype} {base}\n")
    is_sbs = "_sbs" in base
    if not is_sbs and not KEEP_ANAGLYPH:
        print(f"[lossless] skipping anaglyph {base}", flush=True); return
    if is_sbs and LOSSLESS_SBS:
        out = output_video_path[:-4] + ".mkv" if output_video_path.endswith(".mp4") else output_video_path + ".mkv"
        _ffv1_write(arr, fps, out)
        with open(out + ".md5", "w") as fh:
            fh.write(f"{digest}  {tuple(arr.shape)}  fps={float(fps):.6f}\n")
        print(f"[lossless] wrote FFV1 {out} md5(pre-encode)={digest}", flush=True); return
    _orig_write(input_frames, fps, output_video_path)
    print(f"[lossless] wrote mp4v(original) {output_video_path} md5(pre-encode)={digest}", flush=True)

ii.write_video_opencv = _patched_write

# ---------------- hybrid tensor-swap hook, verbatim from minift/xcheck_hybrid_minift.py ----------------
mode, clip, out = sys.argv[1], sys.argv[2], sys.argv[3]
CK = os.environ["MINIFT_CK"]
PREF = ["down_blocks.0.attentions.0.transformer_blocks.0.attn1.", "down_blocks.0.attentions.1.transformer_blocks.0.attn1.",
        "up_blocks.3.attentions.0.transformer_blocks.0.attn1.", "up_blocks.3.attentions.1.transformer_blocks.0.attn1.",
        "up_blocks.3.attentions.2.transformer_blocks.0.attn1."]
raw = torch.load(CK, map_location="cpu", weights_only=False); m = raw.get("model", raw)
e1 = {k: v for k, v in m.items() if any(k.startswith(p) for p in PREF) and "origin_attn" not in k}
if mode == "e1up3": e1 = {k: v for k, v in e1.items() if k.startswith("up_blocks.3")}
if mode == "e1down0": e1 = {k: v for k, v in e1.items() if k.startswith("down_blocks.0")}
print(f"[hybrid] mode={mode} swap-set={len(e1)} tensors ck={CK} LOSSLESS_SBS={int(LOSSLESS_SBS)}", flush=True)
T_SPLIT = 1.0
log = []
_call = ii._Pipe.__call__
def wrapped_call(self, *a, **k):
    unet = self.unet
    if not getattr(unet, "_hybrid_installed", False):
        params = dict(unet.named_parameters())
        origin = {kk: params[kk].detach().clone() for kk in e1}
        e1dev = {kk: e1[kk].to(device=params[kk].device, dtype=params[kk].dtype) for kk in e1}
        nd = sum(int((e1dev[kk] != origin[kk]).any()) for kk in e1)
        print(f"[hybrid] tensors differing from origin in swap-set: {nd}/{len(e1)}", flush=True)
        st = {"cur": None}
        def use(which):
            if st["cur"] == which: return
            src = e1dev if which == "e1" else origin
            with torch.no_grad():
                for kk in e1: params[kk].copy_(src[kk])
            st["cur"] = which
        _fwd = unet.forward
        def fwd(sample, timestep, *fa, **fk):
            t = float(timestep.flatten()[0]) if torch.is_tensor(timestep) else float(timestep)
            high = t > T_SPLIT
            if mode.startswith("e1range:"):
                lo, hi = (float(x) for x in mode.split(":")[1:3]); use("e1" if (lo < t <= hi) else "origin")
            elif mode == "e1high": use("e1" if high else "origin")
            elif mode == "e1low": use("origin" if high else "e1")
            elif mode == "originall": use("origin")
            else: use("e1")
            log.append((round(t, 4), st["cur"]))
            return _fwd(sample, timestep, *fa, **fk)
        unet.forward = fwd; unet._hybrid_installed = True
    return _call(self, *a, **k)
ii._Pipe.__call__ = wrapped_call
ii.run(config="config/0160_overfit_inference_matched.json", unet_state_path=None,
       input_video_path=f"video_data/splatting/{clip}_splatting_results.mp4", save_dir=out)
print("[hybrid] (t, weights) usage:", sorted(collections.Counter(log).items()), flush=True)
print("[hybrid] done", flush=True)
