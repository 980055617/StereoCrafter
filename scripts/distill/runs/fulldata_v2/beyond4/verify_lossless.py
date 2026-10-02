"""Faithfulness proof for the lossless path.  Three separate claims, reported separately."""
import hashlib, math, sys, os
import numpy as np, torch
sys.path.insert(0, "/home/kawa/master_project/StereoCrafter"); os.chdir("/home/kawa/master_project/StereoCrafter")
from decord import VideoReader, cpu
from utils.inpainting import read_and_prepare_video

mkv, mp4v, splat, nlim = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 0

def rd(p, n=0):
    vr = VideoReader(p, ctx=cpu(0)); idx = list(range(len(vr)))
    if n: idx = idx[:n]
    return vr.get_batch(idx).asnumpy()

A = rd(mkv)
print(f"[shape] mkv frames={A.shape}")
pre = open(mkv + ".md5").read().split()[0]
got = hashlib.md5(np.ascontiguousarray(A).tobytes()).hexdigest()
print(f"CLAIM1 writer-is-lossless: pre-encode md5={pre}  decoded md5={got}  MATCH={pre==got}")

# CLAIM 2: left half is the splatting source's top-left quadrant, bit-identical
fps, fl, fw, fm = read_and_prepare_video(splat)
h, w = A.shape[1], A.shape[2] // 2
H, W = fl.shape[2], fl.shape[3]
top, left = (H - h) // 2, (W - w) // 2
src = (fl[: A.shape[0], :, top:top + h, left:left + w] * 255).permute(0, 2, 3, 1).to(dtype=torch.uint8).numpy()
Lh = A[:, :, :w, :]
d = np.abs(src.astype(np.int32) - Lh.astype(np.int32))
print(f"CLAIM2 left-half-passthrough: source={splat} crop=({top},{left}) size={h}x{w} "
      f"BIT_IDENTICAL={bool(d.max()==0)} maxabs={int(d.max())} nonzero={int((d>0).sum())}/{d.size}")

# CLAIM 3: right half vs the shipped mp4v right half = codec error only
B = rd(mp4v, A.shape[0])
Rll = A[:, :, w:, :].astype(np.float64) / 255.0
Rmp = B[:, :, w:, :].astype(np.float64) / 255.0
mse = ((Rll - Rmp) ** 2).mean()
Lmp = B[:, :, :w, :].astype(np.float64) / 255.0
Lll = A[:, :, :w, :].astype(np.float64) / 255.0
mseL = ((Lll - Lmp) ** 2).mean()
sh = lambda X: np.abs(X[:, :, 1:, :] - X[:, :, :-1, :]).mean()
print(f"CLAIM3 lossless-vs-mp4v (n={A.shape[0]} frames of {mp4v}):")
print(f"   right-half PSNR = {10*math.log10(1/max(mse,1e-18)):.2f} dB   maxabs={int(np.abs(A[:,:,w:,:].astype(int)-B[:,:,w:,:].astype(int)).max())}")
print(f"   left-half  PSNR = {10*math.log10(1/max(mseL,1e-18)):.2f} dB   maxabs={int(np.abs(A[:,:,:w,:].astype(int)-B[:,:,:w,:].astype(int)).max())}")
print(f"   sharp(right): lossless={sh(Rll):.5f}  mp4v={sh(Rmp):.5f}   ratio={sh(Rll)/sh(Rmp):.4f}")
