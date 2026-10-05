"""Temporal correspondence check: for every frame i of the splatting video's TL (the left eye the deployed splat used),
find the best-matching frame j of each candidate source (sequential decode, downscaled), and the same for the
train bundle's TL and TR.  usage: frame_align_v1.py <clip> <src1> [<src2> ...]   (CPU)"""
import sys
import numpy as np
from decord import VideoReader, cpu
REPO = "/home/kawa/master_project/StereoCrafter"
clip = sys.argv[1]
srcs = sys.argv[2:]
DS = 8


def load_quads(path, n=None):
    vr = VideoReader(path, ctx=cpu(0))
    f0 = vr[0].asnumpy()
    H, W = f0.shape[0], f0.shape[1]
    vr = VideoReader(path, ctx=cpu(0), width=W // DS, height=H // DS)
    a = vr.get_batch(list(range(len(vr)))).asnumpy().astype(np.float32)
    return a


def load_full(path):
    vr = VideoReader(path, ctx=cpu(0))
    f0 = vr[0].asnumpy()
    H, W = f0.shape[0], f0.shape[1]
    vr = VideoReader(path, ctx=cpu(0), width=W // (DS // 2) // 2 * 2 if False else W // (DS // 2), height=H // (DS // 2))
    return vr.get_batch(list(range(len(vr)))).asnumpy().astype(np.float32)


sp = load_quads(f"{REPO}/video_data/splatting/{clip}_splatting_results.mp4")
h, w = sp.shape[1] // 2, sp.shape[2] // 2
TLs = sp[:, :h, :w]
tr = load_quads(f"{REPO}/video_data/train/{clip}_train.mp4")
ht, wt = tr.shape[1] // 2, tr.shape[2] // 2
TLt, TRt = tr[:, :ht, :wt], tr[:, :ht, wt:]
print(f"{clip}: splat n={len(sp)} quad {h}x{w}; train n={len(tr)} quad {ht}x{wt}")


def psnr(a, b):
    return 10 * np.log10(255 ** 2 / max(((a - b) ** 2).mean(), 1e-9))


def match(A, B, name):
    m = min(A.shape[1], B.shape[1]); n = min(A.shape[2], B.shape[2])
    A = A[:, :m, :n]; B = B[:, :m, :n]
    best = []
    for i in range(len(A)):
        ps = [psnr(A[i], B[j]) for j in range(max(0, i - 25), min(len(B), i + 26))]
        js = list(range(max(0, i - 25), min(len(B), i + 26)))
        k = int(np.argmax(ps))
        best.append((js[k] - i, ps[k], psnr(A[i], B[i]) if i < len(B) else float("nan")))
    off = np.array([b[0] for b in best])
    same = np.array([b[2] for b in best])
    bp = np.array([b[1] for b in best])
    chg = [i for i in range(1, len(off)) if off[i] != off[i - 1]]
    print(f"  {name}: n={len(B)}  offset j-i: first {off[:3].tolist()} last {off[-3:].tolist()} "
          f"unique {sorted(set(off.tolist()))}  changes at i={chg[:12]}  bestPSNR median {np.median(bp):.1f} "
          f"sameIdxPSNR median {np.nanmedian(same):.1f} min {np.nanmin(same):.1f}")


match(TLs, TLt, "train TL vs splat TL")
for s in srcs:
    v = load_quads(f"{REPO}/{s}")
    match(TLs, v, f"{s} vs splat TL")
