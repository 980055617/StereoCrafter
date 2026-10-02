"""Robustness check: does the DC-offset confound change the config-vs-config LPIPS delta?

The splatting input, decoded by decord, sits ~+2/255 brighter than the train GT decoded by
decord.  The tracked mp4v write happens to cancel that; lossless output does not.  The offset
comes from the INPUT, so it should be config-independent and cancel in a delta.  This recomputes
LPIPS with the variant's per-clip global mean matched to the GT's, and prints both.

usage: score_dcmatch.py <clip>=<video> ...
"""
import math, os, sys
import torch, lpips
from decord import VideoReader, cpu
os.chdir("/home/kawa/master_project/StereoCrafter")
STEP = int(os.environ.get("SCORE_STEP", "4"))
def load(p):
    vr = VideoReader(p, ctx=cpu(0))
    f = vr.get_batch(list(range(0, len(vr), STEP))).asnumpy()
    return torch.from_numpy(f).permute(0, 3, 1, 2).float() / 255.
net = lpips.LPIPS(net='alex').cuda().eval()
cur = None
print(f"{'clip/config':30s} {'LPIPS_raw':>10s} {'LPIPS_dcmatch':>14s} {'dcShift/255':>12s} {'meanVar':>8s} {'meanGT':>8s}")
for spec in sys.argv[1:]:
    clip, path = spec.split('=', 1)
    if clip != cur:
        tile = load(f"video_data/train/{clip}_train.mp4"); cur = clip
        H, W = tile.shape[2] // 2, tile.shape[3] // 2
        gtL = tile[:, :, :H, :W]; gtR = tile[:, :, :H, W:2 * W]
    v = load(path); half = v.shape[3] // 2
    L = v[:, :, :, :half]; R = v[:, :, :, half:]
    n = min(len(L), len(gtL)); L, R = L[:n], R[:n]; h, w = L.shape[2], L.shape[3]
    best = ((0, 0), -1)
    for st, rng in ((4, 60), (1, 6)):
        cy, cx = best[0]
        for dy in range(cy - rng, cy + rng + 1, st):
            for dx in range(cx - rng, cx + rng + 1, st):
                t0 = (H - h) // 2 + dy; l0 = (W - w) // 2 + dx
                if t0 < 0 or l0 < 0 or t0 + h > H or l0 + w > W: continue
                m = (L[::6] - gtL[:n:6, :, t0:t0 + h, l0:l0 + w]).pow(2).mean().item()
                p = 10 * math.log10(1 / max(m, 1e-12))
                if p > best[1]: best = ((dy, dx), p)
    (dy, dx), _al = best; t0 = (H - h) // 2 + dy; l0 = (W - w) // 2 + dx
    t = gtR[:n, :, t0:t0 + h, l0:l0 + w]
    shift = (t.mean() - R.mean()).item()
    Rm = (R + shift).clamp(0, 1)
    tot = tot2 = 0.
    with torch.no_grad():
        for i in range(0, n, 4):
            gc = (t[i:i + 4].cuda() * 2 - 1)
            tot += float(net(R[i:i + 4].cuda() * 2 - 1, gc).sum())
            tot2 += float(net(Rm[i:i + 4].cuda() * 2 - 1, gc).sum())
    tag = path.split('/')[-2]
    print(f"{tag[:30]:30s} {tot/n:10.4f} {tot2/n:14.4f} {shift*255:+12.3f} {R.mean().item():8.5f} {t.mean().item():8.5f}")
    print(f"DCROW clip={clip} tag={tag} lpips_raw={tot/n:.6f} lpips_dcmatch={tot2/n:.6f} shift255={shift*255:.4f}")
