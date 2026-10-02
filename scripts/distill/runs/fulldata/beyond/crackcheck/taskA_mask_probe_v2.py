"""v2 of TASK A probe: noise-robust soft-mask bins + left-eye baseline for the black/crack detectors.
Same deployed crop as score_composite.py:28-29. CPU only. Writes outputs/fulldata/beyond/crackcheck/taskA_mask_probe_v2.csv"""
import sys, csv, collections, numpy as np, cv2
from decord import VideoReader, cpu
OUT = "outputs/fulldata/beyond/crackcheck/taskA_mask_probe_v2.csv"; TH, TW = 576, 1024
clips = sys.argv[1:] or ["0042", "0141", "0301", "0259", "0170"]
hdr = ["clip","frame","B_area","m_soft_0.1_0.5","soft_near_B<=2","soft_far_B>2","ncomp","nthin","thin_area/B","B_in_on_latent","black_w_outB","black_L","crack_w_outB","crack_L","crack_w_outB_d<=2","crack_w_outB_d3_8","crack_w_outB_d>8","excess_crack_w_minus_L","bottomrow_B_frac"]
rows = []
def crackdet(g):
    c = np.zeros(g.shape, bool); c[:, 1:-1] = (g[:, :-2] - g[:, 1:-1] > 0.12) & (g[:, 2:] - g[:, 1:-1] > 0.12); return c
for clip in clips:
    vr = VideoReader(f"video_data/splatting/{clip}_splatting_results.mp4", ctx=cpu(0)); n = len(vr)
    idx = list(range(0, n, max(1, n // 8)))[:8]; fr = vr.get_batch(idx).asnumpy()
    h, w = fr.shape[1] // 2, fr.shape[2] // 2; h128, w128 = h // 128 * 128, w // 128 * 128; top, left = (h128 - TH) // 2, (w128 - TW) // 2
    for k, fi in enumerate(idx):
        f = fr[k].astype(np.float32) / 255.0
        m = f[h + top:h + top + TH, left:left + TW].mean(-1); wp = f[h + top:h + top + TH, w + left:w + left + TW]; L = f[top:top + TH, left:left + TW]
        B = (m >= 0.5).astype(np.uint8); Bsum = int(B.sum())
        dist = cv2.distanceTransform(1 - B, cv2.DIST_L2, 3); outB = B == 0
        soft = (m > 0.1) & (m < 0.5)
        ncomp, lab, stats, _ = cv2.connectedComponentsWithStats(B, connectivity=8)
        thin_ids = [i for i in range(1, ncomp) if stats[i, 2] <= 3 or stats[i, 3] <= 3]; thin = np.isin(lab, thin_ids)
        Lup = np.repeat(np.repeat(B[0::8, 0::8], 8, 0), 8, 1)[:TH, :TW].astype(bool)
        black_w = ((wp.max(-1) < 8 / 255) & outB).sum(); black_L = (L.max(-1) < 8 / 255).sum()
        cw = crackdet(wp.mean(-1)) & outB; cL = crackdet(L.mean(-1))
        rows.append([clip, fi, Bsum, int(soft.sum()), int((soft & (dist <= 2)).sum()), int((soft & (dist > 2)).sum()), ncomp - 1, len(thin_ids), thin.sum() / max(Bsum, 1), (B.astype(bool) & Lup).sum() / max(Bsum, 1),
                     int(black_w), int(black_L), int(cw.sum()), int(cL.sum()), int((cw & (dist <= 2)).sum()), int((cw & (dist > 2) & (dist <= 8)).sum()), int((cw & (dist > 8)).sum()), int(cw.sum()) - int(cL.sum()), float(B[-1].mean())])
with open(OUT, "w", newline="") as fh: c = csv.writer(fh); c.writerow(hdr); c.writerows(rows)
by = collections.defaultdict(list)
for r in rows: by[r[0]].append(r)
print("clip | B_area px | soft(0.1,0.5) px [near B<=2 / far>2] | ncomp / nthin | thin_area/B | B_in_on_latent | black warped-outB vs left | crack warped-outB vs left (d<=2 / 3-8 / >8) | excess | bottomrow_B")
for c, rs in by.items():
    a = np.array([r[2:] for r in rs], dtype=np.float64).mean(0)
    print(f"{c} | {a[0]:.0f} | {a[1]:.0f} [{a[2]:.0f} / {a[3]:.0f}] | {a[4]:.0f} / {a[5]:.0f} | {a[6]:.3f} | {a[7]:.3f} | {a[8]:.0f} vs {a[9]:.0f} | {a[10]:.0f} vs {a[11]:.0f} ({a[12]:.0f} / {a[13]:.0f} / {a[14]:.0f}) | {a[15]:+.0f} | {a[16]:.2f}")
