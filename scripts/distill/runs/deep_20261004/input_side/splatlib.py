"""deep_20261004 / input_side lane: shared helpers (CPU).  Nothing tracked is modified.

Geometry facts (verified in this lane, see FEASIBILITY*.txt):
  video_data/splatting/<clip>_splatting_results.mp4 = 2x2 [left | depth_vis(inferno) ; occlusion mask | warped right],
  written by depth_splatting_inference_origin.py (max_disp 20, cv2 mp4v).  The model reads BL/BR, crops each quadrant to
  multiples of 128 from the top-left, then centre-crops 576x1024 (utils/inpainting.py + inpainting_inference.py).
  depth_vis = inferno[floor(255*d)] with d the per-video min-max-normalised DepthCrafter output (the very array splatted).
  deployed disparity  disp = (2d-1)*20 px at full quadrant resolution; flow = -disp (x_right = x_left - disp).
"""
import math
import os

import numpy as np
import torch
from decord import VideoReader, cpu

REPO = "/home/kawa/master_project/StereoCrafter"
TH, TW = 576, 1024
MAX_DISP = 20.0

_LUT = None


def inferno_lut_u8():
    """the uint8 colours the splatting writer produced for each of the 256 inferno entries:
    vis (float colormap entry) -> np.clip(vis*255,0,255).astype(uint8)  (depth_splatting_inference_origin.py)."""
    global _LUT
    if _LUT is None:
        from matplotlib import colormaps
        c = np.asarray(colormaps["inferno"].colors, dtype=np.float64)          # (256,3) float in [0,1]
        # vis_sequence_depth returns float32 numpy (torch tensor of float64 colormap -> .numpy()), the writer does
        # np.clip(video_grid*255.0, 0, 255).astype(np.uint8)
        _LUT = np.clip(c * 255.0, 0, 255).astype(np.uint8)
    return _LUT


LUT_RT = "/mnt/ssd_data/deep_20261004/input_side/feas/lut_roundtrip_3840x2160.npz"


def invert_inferno(rgb_u8, lut="roundtrip"):
    """rgb_u8 [...,3] uint8 -> index k in 0..255 of the nearest LUT colour (squared RGB distance) and that distance.
    lut='roundtrip' (default): the 256 colours as DECODED after the same cv2-mp4v writer + decord reader
    (lut_roundtrip_v1.py; the written colours come back ~2.3/1.5/2.6 darker, which biases a naive inversion low);
    lut='written': the uint8 colours the writer was given."""
    if lut == "roundtrip":
        L = np.load(LUT_RT)["lut_decoded"].astype(np.float64)
    else:
        L = inferno_lut_u8().astype(np.float64)
    lut = L                                                                   # (256,3)
    flat = rgb_u8.reshape(-1, 3).astype(np.float64)
    best_k = np.zeros(flat.shape[0], np.int32)
    best_d = np.full(flat.shape[0], np.inf)
    for k in range(256):
        d = ((flat - lut[k]) ** 2).sum(1)
        m = d < best_d
        best_d[m] = d[m]
        best_k[m] = k
    return best_k.reshape(rgb_u8.shape[:-1]), best_d.reshape(rgb_u8.shape[:-1])


_TREE = {}


def invert_inferno_fast(rgb_u8, lut="roundtrip"):
    """nearest LUT colour (Euclidean RGB) via scipy cKDTree; returns k as FLOAT (exact ties -> mean of the two tied
    indices, instead of invert_inferno's lowest-index rule) and the squared distance."""
    from scipy.spatial import cKDTree
    if lut not in _TREE:
        L = np.load(LUT_RT)["lut_decoded"].astype(np.float64) if lut == "roundtrip" else inferno_lut_u8().astype(np.float64)
        _TREE[lut] = cKDTree(L)
    d, k = _TREE[lut].query(rgb_u8.reshape(-1, 3).astype(np.float64), k=2)
    # exact ties are common (the round-tripped LUT colours are integers): average the tied indices
    tie = np.abs(d[:, 0] ** 2 - d[:, 1] ** 2) < 1e-9
    kf = k[:, 0].astype(np.float32)
    kf[tie] = 0.5 * (k[tie, 0] + k[tie, 1]).astype(np.float32)
    return kf.reshape(rgb_u8.shape[:-1]), (d[:, 0] ** 2).reshape(rgb_u8.shape[:-1])


def window_rows_cols(Hq, Wq):
    """deployed window (top, left) inside a full quadrant: crop to /128 from the top-left, then centre crop."""
    return (Hq // 128 * 128 - TH) // 2, (Wq // 128 * 128 - TW) // 2


def splat_rows(left, disp, mode="bilinear", base=1.414, ss=1, eps=1e-6, wmin=None, upmode="bilinear"):
    """Forward-splat full-width rows horizontally (the deployed ForwardWarpStereo restricted to some rows).

    left  float32 [C,R,W] in [0,1]   (C=3), any device (disp on the same device)
    disp  float32 [R,W]   pixels (positive = shift left, flow = -disp)
    mode  'bilinear'     : Forward_Warp bilinear kernel (2 horizontal taps; y is exact so the vertical taps are 0;
                           a source is dropped unless both taps land inside [0,W), as in the CUDA kernel)
          'nearest_acc'  : each source lands on round(x) with its soft-z weight, ACCUMULATED (deterministic analogue
                           of Forward_Warp 'Nearest', whose CUDA kernel ASSIGNS racily with no z-test)
          'zbuf_bilinear': bilinear taps, but a target only keeps taps of weight >= 0.25 whose disparity is within 1 px
                           of the max disparity landing there (hard z-buffer; Forward_Warp forward_warp_max_motion idea)
    base  soft-z weight base: weight = base**(disp - min(disp))  (deployed 1.414; 1.0 = no z weighting)
    ss    horizontal supersampling: source rows upsampled ss x (upmode bilinear|bicubic, align_corners=False),
          disparity x ss, splat, then weight-normalised box average back to W
    returns warped [C,R,W] float32 (holes = 0), coverage [R,W] = Forward_Warp(ones) (unclamped), wsum [R,W]
    """
    import torch.nn.functional as F
    dev = left.device
    C, R, W = left.shape
    if ss > 1:
        lu = F.interpolate(left[None], size=(R, W * ss), mode=upmode, align_corners=False)[0]
        if upmode == "bicubic":
            lu = lu.clamp(0.0, 1.0)
        du = F.interpolate(disp[None, None], size=(R, W * ss), mode="bilinear", align_corners=False)[0, 0] * ss
        wu, cu, su = splat_rows(lu, du, mode=mode, base=base ** (1.0 / ss), ss=1, eps=eps,
                                wmin=(disp.min() * ss if wmin is None else wmin * ss))
        acc = (wu * su[None]).view(C, R, W, ss).sum(-1)
        ssum = su.view(R, W, ss).sum(-1)
        cov = cu.view(R, W, ss).mean(-1)
        out = acc / ssum.clamp(min=eps)[None]
        return out, cov, ssum
    dmin = disp.min() if wmin is None else wmin
    wgt = torch.pow(torch.tensor(base, dtype=torch.float32, device=dev), disp - dmin)       # [R,W]
    xs = torch.arange(W, dtype=torch.float32, device=dev)[None, :].expand(R, W)
    x = xs + (-disp)                                                                           # float32, as the kernel
    rows = torch.arange(R, device=dev)[:, None].expand(R, W)
    vals = torch.cat([left * wgt[None], wgt[None], torch.ones(1, R, W, device=dev)], 0)        # [C+2,R,W]
    acc = torch.zeros(C + 2, R * W, dtype=torch.float32, device=dev)
    if mode in ("bilinear", "zbuf_bilinear"):
        xf = torch.floor(x)
        xc = xf + 1
        kL = xc - x
        kR = x - xf
        xfi = xf.long()
        ok = (xfi >= 0) & (xfi + 1 < W)
        okL = okR = ok
        if mode == "zbuf_bilinear":
            dq = torch.round(disp * 1000).long()
            dbuf = torch.full((R * W,), -(1 << 40), dtype=torch.long, device=dev)
            for kk, off in ((kL, 0), (kR, 1)):
                m = ok & (kk >= 0.25)
                dbuf.scatter_reduce_(0, (rows * W + xfi + off)[m], dq[m], reduce="amax")
            iL = (rows * W + xfi).clamp(0, R * W - 1)
            iR = (rows * W + xfi + 1).clamp(0, R * W - 1)
            okL = ok & (kL >= 0.25) & ((dbuf[iL] - dq) <= 1000)
            okR = ok & (kR >= 0.25) & ((dbuf[iR] - dq) <= 1000)
        for kk, off, okk in ((kL, 0, okL), (kR, 1, okR)):
            idx = (rows * W + xfi + off)[okk]
            acc.index_add_(1, idx, (vals * kk[None])[:, okk])
    elif mode == "nearest_acc":
        xn = torch.round(x).long()
        ok = (xn >= 0) & (xn < W)
        acc.index_add_(1, (rows * W + xn)[ok], vals[:, ok])
    else:
        raise ValueError(mode)
    acc = acc.view(C + 2, R, W)
    wsum = acc[C]
    out = acc[:C] / wsum.clamp(min=eps)[None]
    return out, acc[C + 1], wsum


def to_u8(x):
    """depth_splatting_inference_origin.py writer: np.clip(v*255.0, 0, 255).astype(np.uint8) (truncation)."""
    return np.clip(x * 255.0, 0, 255).astype(np.uint8)


def psnr_u8(a, b, valid=None):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    if valid is None:
        e = ((a - b) ** 2).mean()
    else:
        v = np.broadcast_to(valid[..., None], a.shape) if a.ndim == valid.ndim + 1 else valid
        e = (((a - b) ** 2) * v).sum() / max(v.sum(), 1)
    return 10 * math.log10(255.0 ** 2 / max(e, 1e-12))


def depthcrafter_proc_size(Hq, Wq, max_res=1024):
    """read_video_frames() of depth_splatting_inference_origin.py: the resolution DepthCrafter ran at."""
    height = round(Hq / 64) * 64
    width = round(Wq / 64) * 64
    if max(height, width) > max_res:
        scale = max_res / max(Hq, Wq)
        height = round(Hq * scale / 64) * 64
        width = round(Wq * scale / 64) * 64
    return height, width


def depth_ls_from_k(k_full, Hq, Wq, iters=60, rows=None):
    """Recover the smooth splatted depth from the decoded inferno indices.

    The splatted depth is F.interpolate(r, (Hq,Wq), bilinear, align_corners=False) of the DepthCrafter output r at
    depthcrafter_proc_size(), min-max normalised (an affine map, which commutes with the interpolation).  We solve
    min_x || up(x) - (k+0.5)/255 ||^2 over the observed rows by conjugate gradient on the normal equations
    (x = low-res map), starting from an area-downsampled guess, and return up(x) (float32 [Hq,Wq]).
    k_full: int array [Hq,Wq] (rows outside `rows` may be anything; only `rows` enter the loss)
    """
    import torch.nn.functional as F
    h, w = depthcrafter_proc_size(Hq, Wq)
    obs = torch.from_numpy((k_full.astype(np.float32) + 0.5) / 255.0)
    m = torch.zeros(Hq, Wq)
    if rows is None:
        m[:] = 1
    else:
        m[rows] = 1

    def up(x):
        return F.interpolate(x[None, None], size=(Hq, Wq), mode="bilinear", align_corners=False)[0, 0]

    def AtA(x):
        x = x.detach().requires_grad_(True)
        y = up(x) * m
        g, = torch.autograd.grad(y, x, grad_outputs=y.detach().clone(), retain_graph=False)
        return g

    def At(b):
        x = torch.zeros(h, w, requires_grad=True)
        y = up(x)
        g, = torch.autograd.grad(y, x, grad_outputs=(b * m))
        return g

    x = F.interpolate(obs[None, None], size=(h, w), mode="area")[0, 0].clone()
    b = At(obs)
    r = b - AtA(x)
    p = r.clone()
    rs = float((r * r).sum())
    for _ in range(iters):
        Ap = AtA(p)
        alpha = rs / max(float((p * Ap).sum()), 1e-30)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = float((r * r).sum())
        if rs_new < 1e-14:
            break
        p = r + (rs_new / rs) * p
        rs = rs_new
    return up(x).detach().numpy().astype(np.float32), x.detach().numpy()


def bilinear_up_matrix(n_in, n_out):
    """dense [n_out, n_in] matrix of F.interpolate(mode='bilinear', align_corners=False) along one axis
    (PyTorch area_pixel_compute_source_index: src = (dst+0.5)*n_in/n_out - 0.5, clamped at 0; i1 = min(i0+1, n_in-1))."""
    U = np.zeros((n_out, n_in), np.float64)
    sc = n_in / n_out
    for i in range(n_out):
        src = (i + 0.5) * sc - 0.5
        if src < 0:
            src = 0.0
        i0 = int(math.floor(src))
        i1 = min(i0 + 1, n_in - 1)
        l1 = src - i0
        U[i, i0] += 1.0 - l1
        U[i, i1] += l1
    return U


class SeparableLS:
    """Closed-form least squares for the splatted depth: d_obs[R0:R1, :] ~= Ur[R0:R1] X Uc^T  (X = DepthCrafter-res map).
    X = pinv(Ur_sub) d_obs pinv(Uc)^T ; returns Ur_win X Uc^T for the window rows.  Exact LS of the CG formulation."""

    def __init__(self, Hq, Wq, R0, R1, top, nrows=576):
        h, w = depthcrafter_proc_size(Hq, Wq)
        Ur = bilinear_up_matrix(h, Hq)
        Uc = bilinear_up_matrix(w, Wq)
        sub = Ur[R0:R1]
        cols = np.where(np.abs(sub).sum(0) > 0)[0]
        self.c0, self.c1 = int(cols.min()), int(cols.max()) + 1
        self.Pr = torch.from_numpy(np.linalg.pinv(sub[:, self.c0:self.c1]))           # [hsub, R1-R0]
        self.Pc = torch.from_numpy(np.linalg.pinv(Uc))                                   # [w, Wq]
        self.Uw = torch.from_numpy(Ur[top:top + nrows, self.c0:self.c1])                 # [nrows, hsub]
        self.Uc = torch.from_numpy(Uc)                                                   # [Wq, w]
        self.R0, self.R1 = R0, R1

    def __call__(self, k_rows):
        """k_rows: float array [R1-R0, Wq] of decoded inferno indices -> depth for the window rows [nrows, Wq] float32"""
        D = torch.from_numpy((k_rows.astype(np.float64) + 0.5) / 255.0)
        X = self.Pr @ D @ self.Pc.T
        return (self.Uw @ X @ self.Uc.T).numpy().astype(np.float32), X.numpy()
