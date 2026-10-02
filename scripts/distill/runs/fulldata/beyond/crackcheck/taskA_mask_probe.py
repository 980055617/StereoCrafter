"""TASK A numeric backing: what the occlusion mask actually contains and what the UNet sees.
CPU only. Per clip, sample frames, apply the deployed crop (quadrant -> //128 -> center 576x1024, same
indexing as scripts/distill/score_composite.py:28-29), then measure:
  - mask value histogram (soft coverage): ==0, (0,0.5), >=0.5
  - binary mask B=(m>=0.5) (pipeline mask_processor do_binarize=True): area, #components, thin components (bbox w<=3 or h<=3)
  - nearest /8 downsample (pipeline _encode_mask_frames F.interpolate default nearest): fraction of B pixels whose latent cell is ON,
    fraction of thin-component pixels whose latent cell is ON
  - warped quadrant: near-black pixels (max channel < 8/255) outside B, split by distance to B (<=2, 3..8, >8 px)
  - horizontal dark-crack detector outside B: pixel darker than both horizontal neighbours by >0.12 (per-channel mean), outside B
Outputs a CSV + per-clip summary to outputs/fulldata/beyond/crackcheck/taskA_mask_probe.csv"""
import sys, os, numpy as np, cv2
from decord import VideoReader, cpu
OUT = "outputs/fulldata/beyond/crackcheck/taskA_mask_probe.csv"
TH, TW = 576, 1024
clips = sys.argv[1:] or ["0042", "0141", "0301", "0259"]
rows = []
hdr = ["clip","frame","m_eq0","m_frac","m_ge05","ncomp","nthin","thin_area_frac_of_B","B_px_in_on_latent","thin_px_in_on_latent","latent_on_cells","black_out_B","black_d<=2","black_d3_8","black_d>8","crack_out_B","crack_d<=2","crack_d3_8","crack_d>8","B_area"]
for clip in clips:
    p = f"video_data/splatting/{clip}_splatting_results.mp4"
    vr = VideoReader(p, ctx=cpu(0)); n = len(vr)
    idx = list(range(0, n, max(1, n // 8)))[:8]
    fr = vr.get_batch(idx).asnumpy()  # T,H2,W2,3 uint8
    h, w = fr.shape[1] // 2, fr.shape[2] // 2
    h128, w128 = h // 128 * 128, w // 128 * 128
    top, left = (h128 - TH) // 2, (w128 - TW) // 2
    for k, fi in enumerate(idx):
        f = fr[k].astype(np.float32) / 255.0
        m = f[h + top:h + top + TH, left:left + TW, :].mean(-1)      # mask quadrant, gray
        wp = f[h + top:h + top + TH, w + left:w + left + TW, :]      # warped quadrant
        eq0 = (m <= 1 / 255).mean(); frac = ((m > 1 / 255) & (m < 0.5)).mean(); ge = (m >= 0.5).mean()
        B = (m >= 0.5).astype(np.uint8)
        ncomp, lab, stats, _ = cv2.connectedComponentsWithStats(B, connectivity=8)
        thin_ids = [i for i in range(1, ncomp) if stats[i, cv2.CC_STAT_WIDTH] <= 3 or stats[i, cv2.CC_STAT_HEIGHT] <= 3]
        thin_mask = np.isin(lab, thin_ids)
        Bsum = B.sum()
        thin_frac = thin_mask.sum() / max(Bsum, 1)
        # nearest /8 : latent cell (i,j) samples pixel (8i, 8j)
        L = B[0::8, 0::8]
        Lup = np.repeat(np.repeat(L, 8, 0), 8, 1)[:TH, :TW]
        b_in_on = (B.astype(bool) & Lup.astype(bool)).sum() / max(Bsum, 1)
        thin_in_on = (thin_mask & Lup.astype(bool)).sum() / max(thin_mask.sum(), 1)
        lat_on = L.mean()
        # distance to B
        dist = cv2.distanceTransform((1 - B).astype(np.uint8), cv2.DIST_L2, 3)
        outB = B == 0
        black = (wp.max(-1) < 8 / 255) & outB
        g = wp.mean(-1)
        crack = np.zeros_like(outB)
        crack[:, 1:-1] = (g[:, :-2] - g[:, 1:-1] > 0.12) & (g[:, 2:] - g[:, 1:-1] > 0.12)
        crack &= outB
        def split(mk):
            return [mk.sum(), (mk & (dist <= 2)).sum(), (mk & (dist > 2) & (dist <= 8)).sum(), (mk & (dist > 8)).sum()]
        rows.append([clip, fi, eq0, frac, ge, ncomp - 1, len(thin_ids), thin_frac, b_in_on, thin_in_on, lat_on] + split(black) + split(crack) + [int(Bsum)])
import csv
with open(OUT, "w", newline="") as fh:
    cw = csv.writer(fh); cw.writerow(hdr); cw.writerows(rows)
print(",".join(hdr))
for r in rows:
    print(",".join(f"{x:.4f}" if isinstance(x, float) else str(x) for x in r))
# per-clip means
print("\n# per-clip means (frac cols) / sums (count cols)")
import collections
by = collections.defaultdict(list)
for r in rows: by[r[0]].append(r)
for c, rs in by.items():
    a = np.array([r[2:] for r in rs], dtype=np.float64)
    mean = a.mean(0)
    print(f"{c}: m_eq0={mean[0]:.4f} m_frac(0,0.5)={mean[1]:.4f} m_ge0.5={mean[2]:.4f} ncomp={mean[3]:.1f} nthin={mean[4]:.1f} thin_area/B={mean[5]:.3f} "
          f"B_px_in_on_latent={mean[6]:.3f} thin_px_in_on_latent={mean[7]:.3f} latent_on={mean[8]:.4f} | black_outB={mean[9]:.0f} (d<=2:{mean[10]:.0f}, 3-8:{mean[11]:.0f}, >8:{mean[12]:.0f}) "
          f"| crack_outB={mean[13]:.0f} (d<=2:{mean[14]:.0f}, 3-8:{mean[15]:.0f}, >8:{mean[16]:.0f}) | B_area_px={mean[17]:.0f} of {TH*TW}")
