#!/usr/bin/env python
"""crack_stats.py -- how much forward-splatting crack / stripe error survives OUTSIDE the disocclusion mask, and whether a
dilated or artefact-aware mask removes it (TASK B of the crackcheck probe, 2026-09-21).

Per clip, every STEP-th frame, in the deployed 576x1024 crop (128-multiple crop from the top-left of each quadrant, then centre
crop -- utils/inpainting.py:read_and_prepare_video + inpainting_inference.py:_center_crop_frames, same indexing as
scripts/distill/score_composite.py):
  video_data/splatting/{clip}_splatting_results.mp4   2x2 tile: TL left, TR depth-vis (inferno LUT of the per-video-normalised
                                                      depth), BL occlusion map (continuous, 1 - splat coverage), BR warped right
  video_data/train/{clip}_train.mp4                    2x2 tile: TL GT left, TR GT right; GT is aligned with the same left-eye
                                                      PSNR search as score_composite.align (coarse +-60 step 4, fine +-6)
  outputs/fulldata/clips/{clip}_origin/{clip}_inpainting_results_sbs.mp4   left | origin right (576x2048)
Regions: hard mask = occlusion map > 0.5 (what score_composite composites with); partial = 16/255 < map <= 0.5 (splat coverage
between 0.5 and 0.94: the under-covered "crack" pixels that the threshold leaves in the non-mask region).
Depth: inverted from the inferno LUT (nearest colour); disparity = (2d-1)*20 (depth_splatting_inference_origin.py:247-248, max_disp
20 verified empirically by backward-warp PSNR peak); gap(x) = 1 + 2*20*(d(x)-d(x+1)) is the target-space spacing of the splats of
source pixels x and x+1 (gap >= 1.5 = stretched, crack-prone; gap > 2 leaves an uncovered pixel = hard mask).
Outputs: crack_stats{TAG}.txt (tables), crack_stats{TAG}.json (all numbers), PNG panels in outputs/fulldata/beyond/crackcheck/.
usage: CUDA_VISIBLE_DEVICES=1 python crack_stats.py [clip ...]       env: STEP (4), TAG ("")
"""
import os, sys, json, math, time
from collections import defaultdict
import numpy as np, cv2, torch, torch.nn.functional as F
from decord import VideoReader, cpu
from scipy.spatial import cKDTree
from matplotlib import colormaps

torch.set_num_threads(6); cv2.setNumThreads(4)
ROOT = "/home/kawa/master_project/StereoCrafter"
OUTD = f"{ROOT}/scripts/distill/runs/fulldata/beyond/crackcheck"; VISD = f"{ROOT}/outputs/fulldata/beyond/crackcheck"
os.makedirs(OUTD, exist_ok=True); os.makedirs(VISD, exist_ok=True)
STEP = int(os.environ.get("STEP", "4")); TAG = os.environ.get("TAG", ""); TH, TW = 576, 1024; MAXDISP = 20.0
TEST = "0042 0052 0125 0128 0141 0147 0170 0204 0225 0251 0259 0301".split(); CLIPS = sys.argv[1:] or TEST + ["0160"]
DILS = [0, 4, 8, 16, 32]; COMP_DILS = [0, 4, 8, 16, 32, 64]; FE = 8
DBINS = [(0, 4), (4, 8), (8, 16), (16, 32), (32, 1e9)]; BINNAMES = ["0-4", "4-8", "8-16", "16-32", ">32"]
ERR_HI = 0.1; CRACK_T = 0.15; CRACK_W = 4; CRACK_L = 8; PART_LO = 16 / 255.; EDGE_T = 3 / 255.; GAP_T = 1.5
dev = "cuda"
LUT = np.array(colormaps["inferno"].colors, dtype=np.float32); TREE = cKDTree(LUT)
import lpips
NET = lpips.LPIPS(net="alex", verbose=False).to(dev).eval()


def uniq_path(p):
    if not os.path.exists(p): return p
    b, e = os.path.splitext(p); i = 1
    while os.path.exists(f"{b}_{i}{e}"): i += 1
    return f"{b}_{i}{e}"


def depth_from_vis(DV):
    flat = DV.reshape(-1, 3); u, inv = np.unique(flat, axis=0, return_inverse=True)
    _, idx = TREE.query(u.astype(np.float32) / 255.)
    return (idx[inv.reshape(-1)].reshape(DV.shape[:2]) / 255.).astype(np.float32)


def load_clip(clip):
    sp = VideoReader(f"{ROOT}/video_data/splatting/{clip}_splatting_results.mp4", ctx=cpu(0))
    gt = VideoReader(f"{ROOT}/video_data/train/{clip}_train.mp4", ctx=cpu(0))
    og = VideoReader(f"{ROOT}/outputs/fulldata/clips/{clip}_origin/{clip}_inpainting_results_sbs.mp4", ctx=cpu(0))
    n = min(len(sp), len(gt), len(og)); idx = list(range(0, n, STEP))
    s0 = sp[0].shape; H, W = s0[0] // 2, s0[1] // 2; h128, w128 = H // 128 * 128, W // 128 * 128; top, left = (h128 - TH) // 2, (w128 - TW) // 2
    g0 = gt[0].shape; GH, GW = g0[0] // 2, g0[1] // 2; cy, cx = (GH - TH) // 2, (GW - TW) // 2; PAD = 72
    wy0, wx0 = max(cy - PAD, 0), max(cx - PAD, 0); wy1, wx1 = min(cy + TH + PAD, GH), min(cx + TW + PAD, GW)
    L = []; DV = []; M = []; WP = []; GL = []; GR = []; OL = []; OR = []
    for i in idx:
        f = sp[i].asnumpy()
        L.append(f[top:top + TH, left:left + TW].copy()); DV.append(f[top:top + TH, W + left:W + left + TW].copy())
        M.append(f[H + top:H + top + TH, left:left + TW].copy()); WP.append(f[H + top:H + top + TH, W + left:W + left + TW].copy())
        g = gt[i].asnumpy(); GL.append(g[wy0:wy1, wx0:wx1].copy()); GR.append(g[wy0:wy1, GW + wx0:GW + wx1].copy())
        o = og[i].asnumpy(); OL.append(o[:TH, :TW].copy()); OR.append(o[:TH, TW:2 * TW].copy())
    st = np.stack
    meta = dict(n_frames=n, idx=idx, H=H, W=W, top=top, left=left, GH=GH, GW=GW, cy=cy, cx=cx, wy0=wy0, wx0=wx0)
    return st(L), st(DV), st(M), st(WP), st(GL), st(GR), st(OL), st(OR), meta


@torch.no_grad()
def align(OL, GL, meta):
    """score_composite.align: origin left eye vs GT left quadrant, offsets relative to the quadrant centre crop."""
    A = torch.from_numpy(OL[::6]).to(dev).float() / 255; B = torch.from_numpy(GL[::6]).to(dev).float() / 255
    wh, ww = B.shape[1], B.shape[2]; best = ((0, 0), -1.0)
    for stp, rng in ((4, 60), (1, 6)):
        cy0, cx0 = best[0]
        for dy in range(cy0 - rng, cy0 + rng + 1, stp):
            for dx in range(cx0 - rng, cx0 + rng + 1, stp):
                t0 = meta["cy"] - meta["wy0"] + dy; l0 = meta["cx"] - meta["wx0"] + dx
                if t0 < 0 or l0 < 0 or t0 + TH > wh or l0 + TW > ww: continue
                m = (A - B[:, t0:t0 + TH, l0:l0 + TW]).pow(2).mean().item(); ps = 10 * math.log10(1 / max(m, 1e-12))
                if ps > best[1]: best = ((dy, dx), ps)
    (dy, dx), ps = best
    return meta["cy"] - meta["wy0"] + dy, meta["cx"] - meta["wx0"] + dx, (dy, dx), ps


def psnr(a, b):
    m = float(((a.astype(np.float32) - b.astype(np.float32)) ** 2).mean()) / 255. ** 2; return 10 * math.log10(1 / max(m, 1e-12))


def thin_cracks(E):
    """Connected components of E that are <= CRACK_W-1 px wide (removed by a CRACK_W x CRACK_W opening) and >= CRACK_L px long."""
    E8 = E.astype(np.uint8); opened = cv2.morphologyEx(E8, cv2.MORPH_OPEN, np.ones((CRACK_W, CRACK_W), np.uint8)); thin = E8 * (1 - opened)
    ncc, lab, st, _ = cv2.connectedComponentsWithStats(thin, connectivity=8)
    if ncc <= 1: return np.zeros(E.shape, bool), 0, 0, 0
    w = st[1:, cv2.CC_STAT_WIDTH]; h = st[1:, cv2.CC_STAT_HEIGHT]; keep = np.maximum(w, h) >= CRACK_L
    cm = np.isin(lab, np.nonzero(keep)[0] + 1)
    return cm, int(keep.sum()), int((keep & (h >= 2 * w)).sum()), int((keep & (w >= 2 * h)).sum())


def span_mask(sel, x0, x1):
    """Boolean (TH,TW): union over selected (y,x) of the integer target span [x0,x1] on row y."""
    diff = np.zeros((TH, TW + 1), np.int32); ys, xx = np.nonzero(sel)
    if len(ys) == 0: return np.zeros((TH, TW), bool)
    a = np.clip(x0[ys, xx], 0, TW - 1).astype(np.int64); b = np.clip(x1[ys, xx], 0, TW - 1).astype(np.int64)
    np.add.at(diff, (ys, a), 1); np.add.at(diff, (ys, b + 1), -1); return np.cumsum(diff, 1)[:, :TW] > 0


def region_acc(acc, k, r, errW, errO, crW, crO, part, crS):
    acc[k + "_cS"] += (crS & r).sum(); acc[k + "_n"] += r.sum(); acc[k + "_sW"] += errW[r].sum(); acc[k + "_sO"] += errO[r].sum()
    acc[k + "_hW"] += (errW[r] > ERR_HI).sum(); acc[k + "_hO"] += (errO[r] > ERR_HI).sum()
    acc[k + "_cW"] += (crW & r).sum(); acc[k + "_cO"] += (crO & r).sum(); acc[k + "_p"] += (part & r).sum()


def frame_stats(G, Wp, O, mc, d, acc):
    hard = mc > 0.5; part = (mc > PART_LO) & ~hard; nm0 = ~hard; hard8 = hard.astype(np.uint8)
    errW = np.abs(Wp - G).mean(-1); errO = np.abs(O - G).mean(-1)
    gG = cv2.cvtColor(G, cv2.COLOR_RGB2GRAY)
    lapW = cv2.Laplacian(cv2.cvtColor(Wp, cv2.COLOR_RGB2GRAY) - gG, cv2.CV_32F, ksize=3)
    lapO = cv2.Laplacian(cv2.cvtColor(O, cv2.COLOR_RGB2GRAY) - gG, cv2.CV_32F, ksize=3)
    crW, ncW, nvW, nhW = thin_cracks((errW > CRACK_T) & nm0); crO, ncO, nvO, nhO = thin_cracks((errO > CRACK_T) & nm0)
    # splat-specific: thin high-error components that coincide (+-1 px) with under-covered (partial) splat pixels
    crS, ncS, nvS, _ = thin_cracks((errW > CRACK_T) & nm0 & (cv2.dilate(part.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0))
    dist = cv2.distanceTransform(1 - hard8, cv2.DIST_L2, 5) if hard.any() else np.full(hard.shape, 1e9, np.float32)
    acc["frames"] += 1; acc["px"] += hard.size; acc["mask_px"] += hard.sum(); acc["part_px"] += part.sum()
    # (a)+(b-laplacian) by mask dilation
    for D in DILS:
        nm = nm0 if D == 0 else (cv2.dilate(hard8, np.ones((2 * D + 1, 2 * D + 1), np.uint8)) == 0)
        k = f"D{D}"; region_acc(acc, k, nm, errW, errO, crW, crO, part, crS)
        acc[k + "_lW"] += float((lapW[nm] ** 2).sum()); acc[k + "_lO"] += float((lapO[nm] ** 2).sum())
    # (b) thin elongated components + partial-coverage pixels
    acc["ncS"] += ncS; acc["nvS"] += nvS; acc["ncW"] += ncW; acc["ncO"] += ncO; acc["nvW"] += nvW; acc["nvO"] += nvO; acc["nhW"] += nhW; acc["nhO"] += nhO
    region_acc(acc, "part", part, errW, errO, crW, crO, part, crS); region_acc(acc, "full", nm0 & ~part, errW, errO, crW, crO, part, crS)
    # (c) distance to nearest hard-mask pixel
    for (lo, hi), nm_ in zip(DBINS, BINNAMES):
        region_acc(acc, "B" + nm_, nm0 & (dist >= lo) & (dist < hi), errW, errO, crW, crO, part, crS)
    # (e) depth-derived regions, mapped to target (right-eye) coordinates
    dxd = d[:, :-1] - d[:, 1:]; gap = 1 + 2 * MAXDISP * dxd
    disp = (2 * d - 1) * MAXDISP; tgt = np.arange(TW, dtype=np.float32)[None, :] - disp
    stretch = span_mask(gap >= GAP_T, np.floor(tgt[:, :-1]) - 1, np.ceil(tgt[:, 1:]) + 1)
    edge = np.abs(dxd) >= EDGE_T
    edge_t = span_mask(edge, np.round(np.minimum(tgt[:, :-1], tgt[:, 1:])), np.round(np.minimum(tgt[:, :-1], tgt[:, 1:]))) | \
             span_mask(edge, np.round(np.maximum(tgt[:, :-1], tgt[:, 1:])), np.round(np.maximum(tgt[:, :-1], tgt[:, 1:])))
    dedge = cv2.dilate(edge_t.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    region_acc(acc, "stretch", stretch & nm0, errW, errO, crW, crO, part, crS)
    region_acc(acc, "dedge", dedge & nm0 & ~stretch, errW, errO, crW, crO, part, crS)
    region_acc(acc, "flat", nm0 & ~stretch & ~dedge, errW, errO, crW, crO, part, crS)
    acc["stretch_frame_px"] += stretch.sum(); acc["stretch_hard"] += (stretch & hard).sum()
    return dict(errW=errW, errO=errO, crW=crW, crO=crO, crS=crS, hard=hard, part=part, stretch=stretch)


@torch.no_grad()
def lpips_variants(G, Wp, O, M, bs=4):
    res = defaultdict(float); n = len(G)
    to = lambda a: torch.from_numpy(a).to(dev).permute(0, 3, 1, 2).float() / 255.
    dil = lambda m, D: F.max_pool2d(m, 2 * D + 1, 1, D) if D > 0 else m
    fea = lambda m: F.avg_pool2d(m, 2 * FE + 1, 1, FE) if FE > 0 else m
    for i in range(0, n, bs):
        g = to(G[i:i + bs]); w = to(Wp[i:i + bs]); o = to(O[i:i + bs]); mc = to(M[i:i + bs]).mean(1, keepdim=True)
        h = (mc > 0.5).float(); p = ((mc > PART_LO) & (mc <= 0.5)).float()
        masks = {f"comp_D{D}": fea(dil(h, D)) for D in COMP_DILS}
        masks["aware_D0"] = fea(torch.maximum(h, p)); masks["aware_D2"] = fea(dil(torch.maximum(h, p), 2)); masks["aware_D4"] = fea(dil(torch.maximum(h, p), 4))
        var = {"origin": o, "warped": w}
        for k, m in masks.items(): var[k] = w * (1 - m) + o * m; res[k + "_gen"] += m.sum().item() / (TH * TW)
        for k, v in var.items():
            res[k + "_lp"] += NET(v * 2 - 1, g * 2 - 1).sum().item(); res[k + "_mae"] += (v - g).abs().mean((1, 2, 3)).sum().item()
    return {k: v / n for k, v in res.items()}


def save_panels(clip, fi, G, Wp, O, fs, d):
    def u8(x): return np.clip(x * 255, 0, 255).astype(np.uint8)
    def errvis(e): return cv2.applyColorMap(np.clip(e * 4 * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)[..., ::-1]
    mv = np.zeros((TH, TW, 3), np.uint8); mv[fs["part"]] = 110; mv[fs["hard"]] = 255; mv[fs["crW"]] = (255, 40, 40); mv[fs["crS"]] = (255, 255, 0)
    dv = cv2.applyColorMap(u8(d), cv2.COLORMAP_INFERNO)[..., ::-1]; sv = dv.copy(); sv[fs["stretch"]] = (0, 255, 0)
    def lab(img, t): img = img.copy(); cv2.putText(img, t, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2); return img
    row1 = np.concatenate([lab(u8(G), "GT right"), lab(u8(Wp), "warped (splat)"), lab(u8(O), "origin")], 1)
    row2 = np.concatenate([lab(errvis(fs["errW"]), "|warped-GT| x4"), lab(errvis(fs["errO"]), "|origin-GT| x4"), lab(mv, "white=hard grey=partial red=thin err(warped) yellow=thin err on partial")], 1)
    row3 = np.concatenate([lab(dv, "depth (from vis)"), lab(sv, "green=stretch gap>=1.5 (target coords)"), lab(np.zeros_like(mv), "")], 1)
    panel = np.concatenate([row1, row2, row3], 0)[::2, ::2]
    cv2.imwrite(uniq_path(f"{VISD}/{clip}_f{fi:03d}_panel.png"), panel[..., ::-1])
    # 1:1 zooms (2x nearest) on the densest partial-coverage window (splat cracks) and the densest thin-error window
    for tag, key in (("part", "part"), ("thin", "crW")):
        dens = cv2.boxFilter(fs[key].astype(np.float32), -1, (192, 192), normalize=True); y, x = np.unravel_index(dens.argmax(), dens.shape)
        y0 = int(np.clip(y - 96, 0, TH - 192)); x0 = int(np.clip(x - 96, 0, TW - 192)); s = (slice(y0, y0 + 192), slice(x0, x0 + 192))
        zoom = np.concatenate([u8(G)[s], u8(Wp)[s], u8(O)[s], errvis(fs["errW"])[s], errvis(fs["errO"])[s], mv[s]], 1)
        zoom = cv2.resize(zoom, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
        cv2.putText(zoom, f"{clip} frame {fi} y{y0} x{x0} ({tag}): GT | warped | origin | errW x4 | errO x4 | mask", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)
        cv2.imwrite(uniq_path(f"{VISD}/{clip}_f{fi:03d}_zoom_{tag}.png"), zoom[..., ::-1])


def derive(acc, lp, meta, al):
    r = {}; f = acc["frames"]; px = acc["px"]
    r["frames"] = int(f); r["mask_pct"] = 100 * acc["mask_px"] / px; r["part_pct"] = 100 * acc["part_px"] / px
    r["align_dy"], r["align_dx"] = al["off"]; r["leftPSNR"] = al["ps"]; r["pipeLeftPSNR"] = al["pipe"]
    def reg(k):
        n = max(acc[k + "_n"], 1)
        return dict(pct=100 * acc[k + "_n"] / px, maeW=acc[k + "_sW"] / n, maeO=acc[k + "_sO"] / n, f01W=100 * acc[k + "_hW"] / n, f01O=100 * acc[k + "_hO"] / n,
                    crackW_pm=1000 * acc[k + "_cW"] / n, crackO_pm=1000 * acc[k + "_cO"] / n, crackS_pm=1000 * acc[k + "_cS"] / n, part_pct=100 * acc[k + "_p"] / n, n=acc[k + "_n"])
    for D in DILS:
        r[f"D{D}"] = reg(f"D{D}"); n = max(acc[f"D{D}_n"], 1); r[f"D{D}"]["lapW"] = acc[f"D{D}_lW"] / n; r[f"D{D}"]["lapO"] = acc[f"D{D}_lO"] / n
    r["cracks"] = dict(ncS=acc["ncS"] / f, vertS=100 * acc["nvS"] / max(acc["ncS"], 1), crackS_pm=1000 * acc["D0_cS"] / max(acc["D0_n"], 1), ncW=acc["ncW"] / f, ncO=acc["ncO"] / f, vertW=100 * acc["nvW"] / max(acc["ncW"], 1), vertO=100 * acc["nvO"] / max(acc["ncO"], 1),
                       horW=100 * acc["nhW"] / max(acc["ncW"], 1), horO=100 * acc["nhO"] / max(acc["ncO"], 1),
                       crackW_pm=1000 * acc["D0_cW"] / max(acc["D0_n"], 1), crackO_pm=1000 * acc["D0_cO"] / max(acc["D0_n"], 1),
                       crackW_in_part=100 * acc["part_cW"] / max(acc["D0_cW"], 1), crackW_share_of_err=100 * acc["D0_cW"] / max(acc["D0_hW"], 1))
    r["part"] = reg("part"); r["full"] = reg("full")
    for b in BINNAMES: r["bin" + b] = reg("B" + b)
    for k in ("stretch", "dedge", "flat"): r[k] = reg(k)
    r["stretch_hard_share"] = 100 * acc["stretch_hard"] / max(acc["stretch_frame_px"], 1)
    r["lpips"] = lp
    return r


def fmt_tables(R, out):
    clips = list(R); test = [c for c in clips if c in TEST]
    def mean(cs, path):
        vals = []
        for c in cs:
            v = R[c]
            for p in path: v = v[p]
            vals.append(v)
        return float(np.mean(vals)) if vals else float("nan")
    rows = clips + [("mean12", test), ("mean13", clips)]
    def cell(c, path, w=7, d=4):
        v = mean(c[1], path) if isinstance(c, tuple) else mean([c], path); return f"{v:{w}.{d}f}"
    def name(c): return c[0] if isinstance(c, tuple) else c
    W = out.write
    W("crack_stats.py  STEP=%d  crop %dx%d  hard mask = occl>0.5, partial = 16/255<occl<=0.5, thin crack = |err|>0.15 comps width<=3 len>=8\n" % (STEP, TH, TW))
    W("err = mean over RGB of |x-GT| in [0,1]; f01 = %% of region pixels with err>0.1; crack_pm = thin-crack pixels per 1000 region pixels; lap = mean Laplacian^2 of gray(x-GT)\n\n")
    W("[0] clip info\n")
    W(f"{'clip':8s} {'frames':>6s} {'mask%':>6s} {'part%':>6s} {'align(dy,dx)':>13s} {'leftPSNR':>8s} {'pipeL':>6s}\n")
    for c in clips:
        r = R[c]; W(f"{c:8s} {r['frames']:6d} {r['mask_pct']:6.2f} {r['part_pct']:6.2f} {str((r['align_dy'], r['align_dx'])):>13s} {r['leftPSNR']:8.2f} {r['pipeLeftPSNR']:6.2f}\n")
    W("\n[A] NON-MASK region (hard mask dilated by D px excluded): MAE and %|err|>0.1, warped-vs-GT (W) and origin-vs-GT (O)\n")
    W(f"{'clip':8s} " + " | ".join(f"D={D:<2d} {'reg%':>5s} {'maeW':>6s} {'maeO':>6s} {'f01W':>5s} {'f01O':>5s}" for D in DILS) + "\n")
    for c in rows:
        W(f"{name(c):8s} " + " | ".join(f"     {cell(c, [f'D{D}', 'pct'], 5, 1)} {cell(c, [f'D{D}', 'maeW'], 6, 4)} {cell(c, [f'D{D}', 'maeO'], 6, 4)} {cell(c, [f'D{D}', 'f01W'], 5, 2)} {cell(c, [f'D{D}', 'f01O'], 5, 2)}" for D in DILS) + "\n")
    W("\n[B1] thin elongated error components (per frame) in the non-mask region; vert% = components with h>=2w; crack_pm = crack pixels per 1000 non-mask px;\n")
    W("     crW_in_part% = share of warped crack pixels that are partial-coverage pixels; crW/err% = share of warped |err|>0.1 pixels that are thin cracks\n")
    W("     S = splat-specific: thin warped-error components lying on (+-1 px) partial-coverage pixels: ncS per frame, vertS% , crS_pm\n")
    W(f"{'clip':8s} {'ncW':>7s} {'ncO':>7s} {'ncS':>6s} {'vertW%':>6s} {'vertO%':>6s} {'vertS%':>6s} {'horW%':>6s} {'horO%':>6s} {'crW_pm':>7s} {'crO_pm':>7s} {'crS_pm':>7s} {'crW_in_part%':>12s} {'crW/err%':>8s}\n")
    for c in rows:
        k = "cracks"; W(f"{name(c):8s} {cell(c, [k, 'ncW'], 7, 1)} {cell(c, [k, 'ncO'], 7, 1)} {cell(c, [k, 'ncS'], 6, 1)} {cell(c, [k, 'vertW'], 6, 1)} {cell(c, [k, 'vertO'], 6, 1)} {cell(c, [k, 'vertS'], 6, 1)} {cell(c, [k, 'horW'], 6, 1)} {cell(c, [k, 'horO'], 6, 1)} {cell(c, [k, 'crackW_pm'], 7, 2)} {cell(c, [k, 'crackO_pm'], 7, 2)} {cell(c, [k, 'crackS_pm'], 7, 2)} {cell(c, [k, 'crackW_in_part'], 12, 1)} {cell(c, [k, 'crackW_share_of_err'], 8, 1)}\n")
    W("\n[B2] high-pass (Laplacian) energy of (x-GT) in the non-mask region, and crack px per 1000, by dilation D\n")
    W(f"{'clip':8s} " + " | ".join(f"D={D:<2d} {'lapW':>7s} {'lapO':>7s} {'crW_pm':>6s} {'crO_pm':>6s}" for D in DILS) + "\n")
    for c in rows:
        W(f"{name(c):8s} " + " | ".join(f"     {cell(c, [f'D{D}', 'lapW'], 7, 5)} {cell(c, [f'D{D}', 'lapO'], 7, 5)} {cell(c, [f'D{D}', 'crackW_pm'], 6, 2)} {cell(c, [f'D{D}', 'crackO_pm'], 6, 2)}" for D in DILS) + "\n")
    W("\n[B3] partial-coverage pixels (16/255<occl<=0.5, NOT in the hard mask) vs fully covered non-mask pixels\n")
    W(f"{'clip':8s} {'part%':>6s} {'maeW@part':>9s} {'maeW@full':>9s} {'f01W@part':>9s} {'f01W@full':>9s} {'maeO@part':>9s} {'maeO@full':>9s} {'f01O@part':>9s} {'f01O@full':>9s} {'crW_pm@part':>11s} {'crW_pm@full':>11s}\n")
    for c in rows:
        W(f"{name(c):8s} {cell(c, ['part', 'pct'], 6, 2)} {cell(c, ['part', 'maeW'], 9, 4)} {cell(c, ['full', 'maeW'], 9, 4)} {cell(c, ['part', 'f01W'], 9, 2)} {cell(c, ['full', 'f01W'], 9, 2)} {cell(c, ['part', 'maeO'], 9, 4)} {cell(c, ['full', 'maeO'], 9, 4)} {cell(c, ['part', 'f01O'], 9, 2)} {cell(c, ['full', 'f01O'], 9, 2)} {cell(c, ['part', 'crackW_pm'], 11, 2)} {cell(c, ['full', 'crackW_pm'], 11, 2)}\n")
    W("\n[C] non-mask error vs distance (px) to the nearest hard-mask pixel  (reg% = % of frame in bin; part% = partial-coverage share within bin)\n")
    W(f"{'clip':8s} {'bin':>5s} {'reg%':>6s} {'maeW':>7s} {'maeO':>7s} {'f01W':>6s} {'f01O':>6s} {'crW_pm':>7s} {'crO_pm':>7s} {'crS_pm':>7s} {'part%':>6s}\n")
    for c in rows:
        for b in BINNAMES:
            k = "bin" + b; W(f"{name(c):8s} {b:>5s} {cell(c, [k, 'pct'], 6, 2)} {cell(c, [k, 'maeW'], 7, 4)} {cell(c, [k, 'maeO'], 7, 4)} {cell(c, [k, 'f01W'], 6, 2)} {cell(c, [k, 'f01O'], 6, 2)} {cell(c, [k, 'crackW_pm'], 7, 2)} {cell(c, [k, 'crackO_pm'], 7, 2)} {cell(c, [k, 'crackS_pm'], 7, 2)} {cell(c, [k, 'part_pct'], 6, 2)}\n")
    W("\n[D] LPIPS vs GT (alex) of the composite = warped outside / origin inside the mask, mask = hard dilated D px, feathered 8 px;\n")
    W("    aware_Dk = (hard OR partial-coverage) dilated k px, feathered 8.  gen% = % of frame taken from origin.  Also MAE of each variant.\n")
    keys = ["origin", "warped"] + [f"comp_D{D}" for D in COMP_DILS] + ["aware_D0", "aware_D2", "aware_D4"]
    W(f"{'LPIPS':8s} " + " ".join(f"{k:>8s}" for k in keys) + "\n")
    for c in rows: W(f"{name(c):8s} " + " ".join(cell(c, ['lpips', k + '_lp'], 8, 4) for k in keys) + "\n")
    W(f"{'gen%':8s} " + " ".join(f"{k:>8s}" for k in keys) + "\n")
    for c in rows: W(f"{name(c):8s} " + " ".join((cell(c, ['lpips', k + '_gen'], 8, 2) if k + '_gen' in R[clips[0]]['lpips'] else f"{'-':>8s}") for k in keys) + "\n")
    W(f"{'MAE':8s} " + " ".join(f"{k:>8s}" for k in keys) + "\n")
    for c in rows: W(f"{name(c):8s} " + " ".join(cell(c, ['lpips', k + '_mae'], 8, 4) for k in keys) + "\n")
    W("\n[E] depth-derived regions in target coords (non-mask only): stretch = splat gap>=1.5 (crack-prone), dedge = +-4 px of a depth step >=3/255 (not stretch), flat = rest\n")
    W("    stretch_hard% = share of the stretch region that is hard mask (i.e. already masked)\n")
    W(f"{'clip':8s} {'region':>8s} {'reg%':>6s} {'maeW':>7s} {'maeO':>7s} {'f01W':>6s} {'f01O':>6s} {'crW_pm':>7s} {'crO_pm':>7s} {'crS_pm':>7s} {'part%':>6s} {'str_hard%':>9s}\n")
    for c in rows:
        for k in ("stretch", "dedge", "flat"):
            W(f"{name(c):8s} {k:>8s} {cell(c, [k, 'pct'], 6, 2)} {cell(c, [k, 'maeW'], 7, 4)} {cell(c, [k, 'maeO'], 7, 4)} {cell(c, [k, 'f01W'], 6, 2)} {cell(c, [k, 'f01O'], 6, 2)} {cell(c, [k, 'crackW_pm'], 7, 2)} {cell(c, [k, 'crackO_pm'], 7, 2)} {cell(c, [k, 'crackS_pm'], 7, 2)} {cell(c, [k, 'part_pct'], 6, 2)} {cell(c, ['stretch_hard_share'], 9, 1) if k == 'stretch' else '':>9s}\n")


def main():
    R = {}; t00 = time.time()
    for clip in CLIPS:
        t0 = time.time(); L, DV, M, WP, GL, GR, OL, OR, meta = load_clip(clip)
        t0g, l0g, off, ps = align(OL, GL, meta); pipe = psnr(OL, L)
        G = np.ascontiguousarray(GR[:, t0g:t0g + TH, l0g:l0g + TW]); acc = defaultdict(float); mid = len(meta["idx"]) // 2
        for j in range(len(G)):
            g = G[j].astype(np.float32) / 255; w = WP[j].astype(np.float32) / 255; o = OR[j].astype(np.float32) / 255
            mc = M[j].astype(np.float32).mean(-1) / 255; d = depth_from_vis(DV[j])
            fs = frame_stats(g, w, o, mc, d, acc)
            if j == mid: save_panels(clip, meta["idx"][j], g, w, o, fs, d)
        lp = lpips_variants(G, WP, OR, M)
        R[clip] = derive(acc, lp, meta, dict(off=off, ps=ps, pipe=pipe))
        r = R[clip]
        print(f"{clip} n={r['frames']} align={off} leftPSNR={ps:.2f} pipeL={pipe:.2f} mask%={r['mask_pct']:.2f} part%={r['part_pct']:.2f} | D0 maeW={r['D0']['maeW']:.4f} maeO={r['D0']['maeO']:.4f} "
              f"f01W={r['D0']['f01W']:.2f} f01O={r['D0']['f01O']:.2f} | cracks/frame W={r['cracks']['ncW']:.1f} O={r['cracks']['ncO']:.1f} | LPIPS origin={lp['origin_lp']:.4f} warped={lp['warped_lp']:.4f} "
              f"D0={lp['comp_D0_lp']:.4f} D8={lp['comp_D8_lp']:.4f} D32={lp['comp_D32_lp']:.4f} aware4={lp['aware_D4_lp']:.4f}  [{time.time() - t0:.0f}s]", flush=True)
        del L, DV, M, WP, GL, GR, OL, OR, G
    txt = uniq_path(f"{OUTD}/crack_stats{TAG}.txt"); js = uniq_path(f"{OUTD}/crack_stats{TAG}.json")
    with open(txt, "w") as f: fmt_tables(R, f)
    with open(js, "w") as f: json.dump(R, f, indent=1, default=float)
    print(f"wrote {txt}\nwrote {js}\npanels in {VISD}\ntotal {time.time() - t00:.0f}s", flush=True)


if __name__ == "__main__":
    main()
