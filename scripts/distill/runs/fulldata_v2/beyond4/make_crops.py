"""3-way GT / origin / g125 crops at 100% zoom on the most texture-rich region.

GT panel uses score_clip.py's offset math VERBATIM:  t0=(H-h)//2+dy ; l0=(W-w)//2+dx
Region is chosen by gradient energy in the GT crop and reused identically in all panels.

usage: make_crops.py <clip> <dy> <dx> <frame_idx> <outdir> <label>=<video> [<label>=<video> ...]
                                                     (first extra label is drawn first, after GT)
"""
import os, sys
import numpy as np
from decord import VideoReader, cpu
from PIL import Image, ImageDraw

os.chdir("/home/kawa/master_project/StereoCrafter")
clip, dy, dx, fidx, outdir = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
panels_spec = [s.split("=", 1) for s in sys.argv[6:]]
os.makedirs(outdir, exist_ok=True)

CROP = 256
def frame(path, i):
    vr = VideoReader(path, ctx=cpu(0))
    i = min(i, len(vr) - 1)
    return vr[i].asnumpy(), len(vr)

# candidate videos (lossless) define h,w
first, nfirst = frame(panels_spec[0][1], fidx)
h, w2 = first.shape[0], first.shape[1]
w = w2 // 2
tile, ngt = frame(f"video_data/train/{clip}_train.mp4", fidx)
H, W = tile.shape[0] // 2, tile.shape[1] // 2
t0 = (H - h) // 2 + dy
l0 = (W - w) // 2 + dx
gt = tile[t0:t0 + h, l0:l0 + w]
print(f"[{clip}] frame={fidx} gt_tile={tile.shape} H,W={H},{W} out={h}x{w} crop_at=({t0},{l0})  gtframes={ngt} vidframes={nfirst}")

g = gt.astype(np.float32).mean(axis=2)
gx = np.abs(np.diff(g, axis=1))[:-1, :]
gy = np.abs(np.diff(g, axis=0))[:, :-1]
energy = gx + gy
ii = np.zeros((energy.shape[0] + 1, energy.shape[1] + 1), np.float64)
ii[1:, 1:] = energy.cumsum(0).cumsum(1)
def box(y, x, s=CROP):
    return ii[y + s, x + s] - ii[y, x + s] - ii[y + s, x] + ii[y, x]
best, by, bx = -1, 0, 0
for y in range(0, energy.shape[0] - CROP, 16):
    for x in range(0, energy.shape[1] - CROP, 16):
        v = box(y, x)
        if v > best: best, by, bx = v, y, x
print(f"[{clip}] texture window y={by} x={bx} size={CROP} meanGrad={best/CROP/CROP:.3f}")

# GT panel first, then each config's RIGHT half, all at the same window
panels = [("GT (real right eye)", gt[by:by + CROP, bx:bx + CROP])]
panels += [(lab, frame(p, fidx)[0][:, w:, :][by:by + CROP, bx:bx + CROP]) for lab, p in panels_spec]

GAP, BAR = 10, 22
Wtot = CROP * len(panels) + GAP * (len(panels) - 1)
img = Image.new("RGB", (Wtot, CROP + BAR), (16, 16, 16))
d = ImageDraw.Draw(img)
for k, (lab, arr) in enumerate(panels):
    x = k * (CROP + GAP)
    img.paste(Image.fromarray(np.ascontiguousarray(arr)), (x, BAR))
    d.text((x + 3, 5), f"{lab}", fill=(240, 240, 240))
p1 = os.path.join(outdir, f"{clip}_f{fidx}_3way_crop_100pct.png")
img.save(p1)
img.resize((Wtot * 3, (CROP + BAR) * 3), Image.NEAREST).save(
    os.path.join(outdir, f"{clip}_f{fidx}_3way_crop_300pct_nearest.png"))

ctx = Image.fromarray(np.ascontiguousarray(frame(panels_spec[0][1], fidx)[0][:, w:, :]))
dd = ImageDraw.Draw(ctx)
dd.rectangle([bx, by, bx + CROP, by + CROP], outline=(255, 40, 40), width=3)
ctx.save(os.path.join(outdir, f"{clip}_f{fidx}_context_origin_with_box.png"))

for lab, arr in panels:
    a = arr.astype(np.float64)
    sh = np.abs(a[:, 1:] - a[:, :-1]).mean() / 255.0
    print(f"    {lab:26s} cropSharp(h)={sh:.5f}")
print(f"[{clip}] wrote {p1}")
