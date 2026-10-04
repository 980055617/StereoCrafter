"""more_20261004 / stripes lane: fill 1-3 px depth-splatting CRACKS in the warped right-eye input
before it reaches the pipeline (and so before the VAE encode and the CLIP image embedding).

Definitions (fixed before any render or score, see PREREG.txt):
  hole pixel   = mask >= 0.5, the SAME binarisation the pipeline's mask_processor applies
                 (VaeImageProcessor(do_binarize=True): <0.5 -> 0, else 1).  The mask quadrant of the
                 splatting mp4 is lossy (h264), so it is continuous; 0.5 is the pipeline's own cut.
  crack pixel  = a hole pixel that belongs to a HORIZONTAL run of hole pixels of length <= MAXW
                 (default 3) in its row.  Forward splatting with horizontal disparity opens gaps along
                 rows, so the row run length is the crack width.  Runs > MAXW are disocclusion AREAS and
                 are left exactly as they are (black, masked) for the model to inpaint.
  fills (applied to crack pixels only; every other pixel is bit-identical to the input):
    rowlin  row-wise linear interpolation between the nearest non-hole pixel to the left (s-1) and to
            the right (e) of the run [s, e); at a region border the one existing neighbour is copied.
    telea   cv2.inpaint(uint8 RGB, crack mask, inpaintRadius=3, cv2.INPAINT_TELEA) per frame.  The input
            frames are exact multiples of 1/255 (decord uint8 / 255), so the uint8 round trip is lossless.
  mask option:
    keep    the mask channel handed to the pipeline is unchanged
    shrink  the mask is set to 0 at the filled crack pixels, i.e. only hole runs wider than MAXW stay
            masked (variant (c))
All work is done on the deployed centre-crop window plus a MARGIN (default 16 px) on each side, so a run
that crosses the window border is measured and filled with its real neighbours; main() then crops the
window exactly as before.
"""
import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


def crack_mask(holes: np.ndarray, maxw: int = 3):
    """holes: bool [..., W].  Returns (crack bool same shape, run arrays (row, start, end) of crack runs)."""
    shp = holes.shape
    x = holes.reshape(-1, shp[-1])
    n = x.shape[0]
    pad = np.zeros((n, 1), np.int8)
    d = np.diff(np.concatenate([pad, x.astype(np.int8), pad], axis=1), axis=1)
    rs, cs = np.nonzero(d == 1)      # run starts   (row-major order)
    re_, ce = np.nonzero(d == -1)    # run ends, exclusive (same order, one end per start)
    assert len(rs) == len(re_) and np.array_equal(rs, re_), "run start/end pairing failed"
    lens = ce - cs
    keep = lens <= maxw
    out = np.zeros_like(x, dtype=bool)
    for L in range(1, maxw + 1):
        sel = keep & (lens == L)
        for k in range(L):
            out[rs[sel], cs[sel] + k] = True
    return out.reshape(shp), (rs[keep], cs[keep], ce[keep]), lens


def fill_rowlin(img: np.ndarray, runs, W: int):
    """img float32 [N, W, C] (rows flattened), runs = (row, start, end) of crack runs.  In place."""
    r, s, e = runs
    if len(r) == 0:
        return img
    has_l = s > 0
    has_r = e < W
    li = np.where(has_l, s - 1, e)            # no left neighbour  -> copy the right one
    ri = np.where(has_r, e, s - 1)            # no right neighbour -> copy the left one
    assert np.all(has_l | has_r), "a crack run spans the whole row"
    Lv = img[r, li]                            # [k, C]
    Rv = img[r, ri]
    n = (e - s)
    for L in np.unique(n):
        sel = n == L
        for k in range(int(L)):
            w = (k + 1) / (L + 1)
            img[r[sel], s[sel] + k] = (Lv[sel] * (1 - w) + Rv[sel] * w).astype(img.dtype)
    return img


def fill_telea(u8_hwc: np.ndarray, crack_hw: np.ndarray, radius: int = 3):
    assert cv2 is not None
    if not crack_hw.any():
        return u8_hwc
    return cv2.inpaint(np.ascontiguousarray(u8_hwc), crack_hw.astype(np.uint8) * 255, radius, cv2.INPAINT_TELEA)


def process(warped_tchw, mask_t1hw, top, left, th, tw, mode, mask_mode, maxw=3, margin=16, log=None):
    """Fill cracks in-place on torch tensors warped [T,3,H,W] and mask [T,1,H,W] (float, [0,1]),
    restricted to the window [top:top+th, left:left+tw] plus `margin`.  Returns a stats dict."""
    import torch
    T, C, H, W = warped_tchw.shape
    y0, y1 = max(0, top - margin), min(H, top + th + margin)
    x0, x1 = max(0, left - margin), min(W, left + tw + margin)
    # window coordinates inside the margin region (for the window-only statistics)
    wy0, wx0 = top - y0, left - x0
    reg_w = warped_tchw[:, :, y0:y1, x0:x1].permute(0, 2, 3, 1).contiguous().numpy()   # [T,h,w,3] float32
    reg_m = mask_t1hw[:, 0, y0:y1, x0:x1].contiguous().numpy()                          # [T,h,w]
    holes = reg_m >= 0.5
    crack, runs, lens = crack_mask(holes, maxw)
    h, w = holes.shape[1], holes.shape[2]
    win = (slice(None), slice(wy0, wy0 + th), slice(wx0, wx0 + tw))
    st = dict(mode=mode, mask_mode=mask_mode, maxw=maxw, margin=margin, region=[y0, y1, x0, x1],
              window=[top, top + th, left, left + tw],
              hole_frac_window=float(holes[win].mean()), crack_frac_window=float(crack[win].mean()),
              crack_share_of_holes_window=float(crack[win].sum() / max(1, holes[win].sum())),
              n_crack_runs_region=int(len(runs[0])),
              runlen_hist_region={str(k): int((lens == k).sum()) for k in range(1, 9)},
              runlen_gt8_region=int((lens > 8).sum()))
    before = reg_w.copy()
    if mode == "rowlin":
        flat = reg_w.reshape(T * h, w, C)
        fill_rowlin(flat, runs, w)
        reg_w = flat.reshape(T, h, w, C)
    elif mode == "telea":
        for t in range(T):
            if not crack[t].any():
                continue
            u8 = np.rint(reg_w[t] * 255.0).astype(np.uint8)
            assert np.array_equal(u8.astype(np.float32) / 255.0, reg_w[t]), "input is not exact k/255"
            out = fill_telea(u8, crack[t], 3)
            reg_w[t] = out.astype(np.float32) / 255.0
    elif mode == "none":
        pass
    else:
        raise ValueError(mode)
    # statistics, frame by frame (bounded memory: the 4400x4400 clips already hold ~25 GB at this point)
    n_chg_win = n_chg_out = 0
    s_abs = s_bef = s_aft = 0.0
    n_cw = 0
    for t in range(T):
        chg = np.any(reg_w[t] != before[t], axis=-1)
        cw = np.zeros_like(chg)
        cw[wy0:wy0 + th, wx0:wx0 + tw] = crack[t, wy0:wy0 + th, wx0:wx0 + tw]
        n_chg_win += int(chg[wy0:wy0 + th, wx0:wx0 + tw].sum())
        n_chg_out += int((chg & ~crack[t]).sum())
        if cw.any():
            s_abs += float(np.abs(reg_w[t][cw] - before[t][cw]).sum())
            s_bef += float(before[t][cw].sum())
            s_aft += float(reg_w[t][cw].sum())
            n_cw += int(cw.sum()) * C
    st["changed_px_window"] = n_chg_win
    st["changed_outside_crack_region"] = n_chg_out
    st["mean_abs_change_on_crack_window"] = s_abs / max(1, n_cw)
    st["mean_input_on_crack_window_before"] = s_bef / max(1, n_cw)
    st["mean_input_on_crack_window_after"] = s_aft / max(1, n_cw)
    del before
    if mode != "none":
        assert st["changed_outside_crack_region"] == 0, "fill touched a non-crack pixel"
        warped_tchw[:, :, y0:y1, x0:x1] = torch.from_numpy(reg_w).permute(0, 3, 1, 2)
    if mask_mode == "shrink":
        m2 = reg_m.copy()
        m2[crack] = 0.0
        mask_t1hw[:, 0, y0:y1, x0:x1] = torch.from_numpy(m2)
        st["mask_px_zeroed_window"] = int(crack[win].sum())
        st["hole_frac_window_after_shrink"] = float((m2 >= 0.5)[win].mean())
    elif mask_mode != "keep":
        raise ValueError(mask_mode)
    return st, crack[win]
