"""Shared helpers for the skeptic lane (deep_20261004).  CPU only.  Nothing tracked is modified; renders and
bundles are only read.  Definitions follow PREREG.txt in this directory."""
import json
import math
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as Fnn
from decord import VideoReader, cpu

REPO = "/home/kawa/master_project/StereoCrafter"
os.chdir(REPO)
TH, TW = 576, 1024
TEST = ["0042", "0052", "0125", "0128", "0141", "0147", "0170", "0204", "0225", "0251", "0259", "0301"]
AVP = ["0042", "0052", "0125", "0128", "0141", "0147"]
IPH = ["0170", "0204", "0225", "0251", "0259", "0301"]
NEARMONO = ["0204", "0225", "0251", "0259", "0301"]
DEV = ["0040", "0082", "0091", "0184", "0245", "0268", "0311", "0351"]
ROWS = json.load(open("scripts/distill/runs/ays_20261004/robust/ROWS_pass1.json"))
ROWLAB = {"origin": "origin_ll", "AYS8": "AYS8_origin_g101", "deliverable": "mstudent2_step800_deliv_ll",
          "s25": "s25_ll", "T5nat": "deliv_g100_T5nat", "T5pad": "deliv_g100_T5pad"}
# outer box (relative to the deployed window) read from the TL / TR quadrants of the train tile and splat tile
BOX_T, BOX_B, BOX_L, BOX_R = 64, 64, 400, 200
# RAFT crop margin (relative to the window)
MY, MX = 32, 128


def family(clip):
    c = int(clip)
    return "AVP" if c <= 159 else ("iPhone" if c <= 309 else "INVALID")


def regjson(clip):
    p = ("outputs/more_20261004/eval_robustness/score_v1_wide/0125.json" if clip == "0125"
         else f"outputs/more_20261004/eval_robustness/score_v1/{clip}.json")
    return json.load(open(p)), p


def row_path(clip, row):
    return ROWS["cells"][clip][ROWLAB[row]]["path"]


def check_valid_train_bundle(clip):
    real = os.path.realpath(f"video_data/train/{clip}_train.mp4")
    assert int(clip) < 310, f"{clip}: id >= 0310 has no valid GT"
    assert "train_leftGT_broken" not in real, f"{clip}: bundle links into train_leftGT_broken"
    return real


def decode_boxes(path, idxs, boxes, chunk=8):
    """boxes: dict name -> (quadrant 'TL'|'TR'|'BL'|'BR', y0, y1, x0, x1) in quadrant coordinates.
    Returns dict name -> uint8 array [len(idxs), y1-y0, x1-x0, 3] and (H, W) of a quadrant."""
    vr = VideoReader(path, ctx=cpu(0))
    f0 = vr[0].asnumpy()
    H, W = f0.shape[0] // 2, f0.shape[1] // 2
    out = {k: np.empty((len(idxs), b[2] - b[1], b[4] - b[3], 3), np.uint8) for k, b in boxes.items()}
    for s in range(0, len(idxs), chunk):
        part = idxs[s:s + chunk]
        bt = vr.get_batch(part).asnumpy()
        for j, fr in enumerate(bt):
            for k, (q, y0, y1, x0, x1) in boxes.items():
                oy = 0 if q in ("TL", "TR") else H
                ox = 0 if q in ("TL", "BL") else W
                assert 0 <= y0 and y1 <= H and 0 <= x0 and x1 <= W, (k, q, y0, y1, x0, x1, H, W)
                out[k][s + j] = fr[oy + y0:oy + y1, ox + x0:ox + x1]
        del bt
    return out, (H, W), len(vr)


def load_clip(clip, frames, want_splat=True):
    """Loads TL/TR boxes of the train tile and BR/BL/TL boxes of the splat tile for the given frame indices."""
    check_valid_train_bundle(clip)
    js, jp = regjson(clip)
    t0, l0 = js["window"]
    tb = {"TL": ("TL", t0 - BOX_T, t0 + TH + BOX_B, l0 - BOX_L, l0 + TW + BOX_R),
          "TR": ("TR", t0 - BOX_T, t0 + TH + BOX_B, l0 - BOX_L, l0 + TW + BOX_R)}
    tr, (H, W), n_t = decode_boxes(f"video_data/train/{clip}_train.mp4", frames, tb)
    d = dict(clip=clip, js=js, jpath=jp, t0=t0, l0=l0, H=H, W=W, frames=list(frames),
             TLtrain=tr["TL"], TR=tr["TR"])
    if want_splat:
        sb = {"BR": ("BR", t0 - MY, t0 + TH + MY, l0 - MX, l0 + TW + MX),
              "BL": ("BL", t0 - MY, t0 + TH + MY, l0 - MX, l0 + TW + MX),
              "TL": ("TL", t0 - BOX_T, t0 + TH + BOX_B, l0 - BOX_L, l0 + TW + BOX_R)}
        sp, (Hs, Ws), n_s = decode_boxes(f"video_data/splatting/{clip}_splatting_results.mp4", frames, sb)
        assert (Hs, Ws) == (H, W)
        d.update(BRext=sp["BR"], BLext=sp["BL"], TLsplat=sp["TL"])
    return d


def box_crop(arr, dy, dx, h=TH, w=TW, my=0, mx=0):
    """Crop from a BOX array (window at (BOX_T, BOX_L)) the window shifted by (dy, dx), extended by (my, mx)."""
    y0 = BOX_T + dy - my
    x0 = BOX_L + dx - mx
    assert y0 >= 0 and x0 >= 0 and y0 + h + 2 * my <= arr.shape[-3] and x0 + w + 2 * mx <= arr.shape[-2], \
        (dy, dx, my, mx, arr.shape)
    return arr[..., y0:y0 + h + 2 * my, x0:x0 + w + 2 * mx, :]


def reg_shift(js, fi, variant="REG_FRAME"):
    if variant == "UNREG":
        return 0, 0
    if variant == "REG_CLIP":
        return js["reg"]["clip_ddy"], js["reg"]["clip_ddx"]
    if variant == "REG_FRAME":
        return int(js["reg"]["smooth_ddy"][fi]), int(js["reg"]["smooth_ddx"][fi])
    if variant == "REG_FRAME_RAW":
        return int(js["reg"]["raw_ddy"][fi]), int(js["reg"]["raw_ddx"][fi])
    raise KeyError(variant)


def render_right(path, frames):
    vr = VideoReader(path, ctx=cpu(0))
    b = vr.get_batch(frames).asnumpy()
    half = b.shape[2] // 2
    assert b.shape[1:3] == (TH, 2 * TW), b.shape
    return b[:, :, :half].copy(), b[:, :, half:].copy()


def t01(a):
    """uint8 or float [n,h,w,3] -> torch float [n,3,h,w] in [0,1]"""
    if a.dtype == np.uint8:
        return torch.from_numpy(np.ascontiguousarray(a)).permute(0, 3, 1, 2).float() / 255.
    return torch.from_numpy(np.ascontiguousarray(a)).permute(0, 3, 1, 2).float()


def q8(x):
    """float [0,1] numpy -> uint8 (clip + round), the way a render would be stored."""
    return np.clip(np.round(x * 255.0), 0, 255).astype(np.uint8)


class Lp:
    def __init__(self, net="alex"):
        import lpips
        self.net = lpips.LPIPS(net=net, verbose=False).eval()
        self.sp = lpips.LPIPS(net=net, spatial=True, verbose=False).eval()

    @torch.no_grad()
    def __call__(self, a_u8, b_u8, bs=4):
        """per-frame LPIPS list, score_clip_ll batching (batches of 4, inputs *2-1)."""
        A, B = t01(a_u8), t01(b_u8)
        out = []
        for i in range(0, len(A), bs):
            out += [float(x) for x in self.net(A[i:i + bs] * 2 - 1, B[i:i + bs] * 2 - 1).view(-1)]
        return out

    @torch.no_grad()
    def spatial(self, a_u8, b_u8, bs=2):
        A, B = t01(a_u8), t01(b_u8)
        out = []
        for i in range(0, len(A), bs):
            out.append(self.sp(A[i:i + bs] * 2 - 1, B[i:i + bs] * 2 - 1)[:, 0].numpy())
        return np.concatenate(out, 0)


def psnr_u8(a, b, m=None):
    a = a.astype(np.float64); b = b.astype(np.float64)
    if m is None:
        e = ((a - b) ** 2).mean()
    else:
        mm = np.broadcast_to(m[..., None], a.shape)
        e = (((a - b) ** 2) * mm).sum() / max(mm.sum(), 1)
    return 10 * math.log10(255.0 ** 2 / max(e, 1e-12))


def sharp_score(a_u8):
    """score_clip_ll 'sharp': mean |horizontal diff| over RGB in [0,1]."""
    x = a_u8.astype(np.float32) / 255.
    return float(np.abs(x[:, :, 1:] - x[:, :, :-1]).mean())


def luma(a):
    x = a.astype(np.float32)
    if a.dtype == np.uint8:
        x = x / 255.
    return 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]


# ------------------------------------------------------------------ spectra
_HANN = {}


def band_power(y, bands=((1 / 32, 1 / 16), (1 / 16, 1 / 8), (1 / 8, 1 / 4), (1 / 4, 0.5001))):
    """Hann-windowed radially binned power of a 2-D luma image (mean removed). Returns list per band."""
    h, w = y.shape
    if (h, w) not in _HANN:
        wy = np.hanning(h)[:, None]; wx = np.hanning(w)[None, :]
        fy = np.fft.fftfreq(h)[:, None]; fx = np.fft.fftfreq(w)[None, :]
        _HANN[(h, w)] = (wy * wx, np.sqrt(fy ** 2 + fx ** 2))
    win, rad = _HANN[(h, w)]
    z = (y - y.mean()) * win
    P = np.abs(np.fft.fft2(z)) ** 2
    return [float(P[(rad >= lo) & (rad < hi)].mean()) for lo, hi in bands]


def immerkaer_sigma(y):
    """Immerkaer (1996) fast noise estimate on a 2-D luma image in [0,1]; returns sigma in 8-bit units."""
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    c = cv2.filter2D(y.astype(np.float32), -1, k, borderType=cv2.BORDER_REFLECT)[1:-1, 1:-1]
    h, w = y.shape
    s = np.sum(np.abs(c)) * math.sqrt(0.5 * math.pi) / (6 * (w - 2) * (h - 2))
    return float(s * 255.0)


def lab_mean(a_u8, m=None):
    """mean CIELAB of an image or a stack [..., h, w, 3] (optionally over mask m [..., h, w])."""
    x = a_u8.astype(np.float32).reshape(-1, 1, 3) / 255.
    lab = cv2.cvtColor(x, cv2.COLOR_RGB2LAB).reshape(-1, 3)
    if m is None:
        return lab.mean(0)
    return lab[np.asarray(m).reshape(-1)].mean(0)


# ------------------------------------------------------------------ colour fit
def fit_affine(src, dst):
    """least squares dst ~ [src,1] @ M, src/dst float [N,3] -> M [4,3]"""
    X = np.concatenate([src, np.ones((len(src), 1), src.dtype)], 1).astype(np.float64)
    M, *_ = np.linalg.lstsq(X, dst.astype(np.float64), rcond=None)
    return M


def apply_affine(img_u8, M):
    x = img_u8.astype(np.float64) / 255.
    sh = x.shape
    y = np.concatenate([x.reshape(-1, 3), np.ones((x.size // 3, 1))], 1) @ M
    return q8(y.reshape(sh))


# ------------------------------------------------------------------ flows / warps
_RAFT = None


def raft():
    global _RAFT
    if _RAFT is None:
        from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
        _RAFT = raft_large(weights=Raft_Large_Weights.C_T_SKHT_V2).eval()
    return _RAFT


@torch.no_grad()
def flow(a_u8, b_u8, iters=12):
    """RAFT flow a -> b for one pair of uint8 [h,w,3] (h,w divisible by 8). Returns float32 [h,w,2] (dx, dy)."""
    A = t01(a_u8[None]) * 2 - 1
    B = t01(b_u8[None]) * 2 - 1
    f = raft()(A, B, num_flow_updates=iters)[-1][0]
    return f.permute(1, 2, 0).numpy().astype(np.float32)


def warp(img, fl, mode="bicubic"):
    """backward warp: out(x) = img(x + fl(x)); img uint8/float [h,w,3] (or [h,w]), fl [h,w,2].  float out [0,1]."""
    x = img.astype(np.float32)
    if img.dtype == np.uint8:
        x = x / 255.
    if x.ndim == 2:
        x = x[..., None]
    h, w = fl.shape[:2]
    gy, gx = np.mgrid[0:h, 0:w].astype(np.float32)
    sx = (gx + fl[..., 0]) / (w - 1) * 2 - 1
    sy = (gy + fl[..., 1]) / (h - 1) * 2 - 1
    grid = torch.from_numpy(np.stack([sx, sy], -1))[None]
    t = torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1)[None]
    o = Fnn.grid_sample(t, grid, mode=mode, padding_mode="border", align_corners=True)[0].permute(1, 2, 0).numpy()
    return np.clip(o, 0, 1).squeeze(-1) if o.shape[-1] == 1 else np.clip(o, 0, 1)


def consistency(f12, f21):
    """True where the forward-backward check passes and x+f12 stays inside frame 2 (Sundaram et al. 2010)."""
    h, w = f12.shape[:2]
    gy, gx = np.mgrid[0:h, 0:w].astype(np.float32)
    tx, ty = gx + f12[..., 0], gy + f12[..., 1]
    inb = (tx >= 0) & (tx <= w - 1) & (ty >= 0) & (ty <= h - 1)
    # warp() clips to [0,1]; flows need an unclipped sampler:
    t = torch.from_numpy(np.ascontiguousarray(f21)).permute(2, 0, 1)[None]
    grid = torch.from_numpy(np.stack([tx / (w - 1) * 2 - 1, ty / (h - 1) * 2 - 1], -1))[None]
    f21w = Fnn.grid_sample(t, grid, mode="bilinear", padding_mode="border", align_corners=True)[0].permute(1, 2, 0).numpy()
    s = f12 + f21w
    lhs = (s ** 2).sum(-1)
    rhs = 0.01 * ((f12 ** 2).sum(-1) + (f21w ** 2).sum(-1)) + 0.5
    return (lhs <= rhs) & inb


def inpaint_holes(img_u8, hole):
    return cv2.inpaint(img_u8, hole.astype(np.uint8) * 255, 3, cv2.INPAINT_TELEA)


def shift_bicubic(img_u8, dy, dx):
    h, w = img_u8.shape[:2]
    fl = np.zeros((h, w, 2), np.float32)
    fl[..., 0] = dx; fl[..., 1] = dy
    return q8(warp(img_u8, fl))


def gauss(x, s):
    if s <= 0:
        return x
    return cv2.GaussianBlur(x, (0, 0), s, borderType=cv2.BORDER_REFLECT)


def best_shift_psnr(src_box, ref, ys, xs, step=4, refine=3):
    """integer (dy,dx) maximising PSNR between box_crop(src_box,dy,dx) and ref (uint8 [TH,TW,3]); coarse-to-fine."""
    ref32 = ref.astype(np.float32)

    def ps(dy, dx):
        c = box_crop(src_box, dy, dx).astype(np.float32)
        return -float(((c - ref32) ** 2).mean())
    best = None
    for dy in range(ys[0], ys[1] + 1, max(1, step // 2)):
        for dx in range(xs[0], xs[1] + 1, step):
            v = ps(dy, dx)
            if best is None or v > best[0]:
                best = (v, dy, dx)
    _, cy, cx = best
    for dy in range(max(ys[0], cy - refine), min(ys[1], cy + refine) + 1):
        for dx in range(max(xs[0], cx - refine), min(xs[1], cx + refine) + 1):
            v = ps(dy, dx)
            if v > best[0]:
                best = (v, dy, dx)
    mse = -best[0]
    return best[1], best[2], 10 * math.log10(255.0 ** 2 / max(mse, 1e-12))
