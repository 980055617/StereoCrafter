"""Round-trip the 256 inferno uint8 colours through the SAME writer path as the splatting video
(cv2.VideoWriter mp4v, RGB->BGR) and decord, as flat 64x64 patches, to calibrate the inverse LUT.
usage: lut_roundtrip_v1.py <out_npz> <W> <H>    (CPU)"""
import sys, os
import numpy as np
import cv2
from decord import VideoReader, cpu
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import splatlib as S
out, Wv, Hv = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
assert not os.path.exists(out), out
lut = S.inferno_lut_u8()
P = 64
nx = Wv // P
img = np.zeros((Hv, Wv, 3), np.uint8)
cent = []
for k in range(256):
    y, x = (k // nx) * P, (k % nx) * P
    img[y:y + P, x:x + P] = lut[k]
    cent.append((y + P // 2, x + P // 2))
tmp = out.replace(".npz", "_tmp.mp4")
vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (Wv, Hv))
for _ in range(5):
    vw.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
vw.release()
vr = VideoReader(tmp, ctx=cpu(0))
dec = vr[2].asnumpy()
got = np.array([dec[y - 8:y + 8, x - 8:x + 8].reshape(-1, 3).mean(0) for y, x in cent])
d = got - lut.astype(np.float64)
print(f"decoded-minus-written colour: mean {d.mean(0).round(2)}  mean|.| {np.abs(d).mean(0).round(2)}  max|.| {np.abs(d).max(0).round(1)}")
np.savez(out, lut_written=lut, lut_decoded=got)
os.remove(tmp)
print("wrote", out)
