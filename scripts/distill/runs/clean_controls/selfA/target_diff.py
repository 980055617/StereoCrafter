"""CONTROL A pre-training readout: what did the old P1-null control actually train toward?
Compares the RIGHT half (= the training target) of
  LL : outputs/beyond4_lossless/clips/0301_origin_ll/0301_inpainting_results_sbs.mkv   (FFV1, decord bit-exact)
  MP : outputs/fulldata_v2/clips/0301_origin/0301_inpainting_results_sbs.mp4           (cv2 mp4v, the old target)
Both hold the SAME pre-encode array (md5 2e533d7755c950d2fc95043f6fb0a51d, beyond4/FAITHFULNESS.txt CHECK 2), so every
difference reported here is the codec.  Also reports the LEFT-half registration of both files against the splatting input's
TL crop (LL must be bit-identical) and against the train tile's TL crop (what the trainer's original assertion sees), and the
cond (BR) mismatch between the splatting file (what produced the target) and the train tile (what the trainer feeds).
CPU only.  usage: python target_diff.py [frame ...]   (default frames 30 75 120; the all-frame mean is always appended)
"""
import os, sys, math, torch, torch.nn.functional as F
REPO = "/home/kawa/master_project/StereoCrafter"; os.chdir(REPO)
from decord import VideoReader, cpu
CLIP = "0301"
LL = f"outputs/beyond4_lossless/clips/{CLIP}_origin_ll/{CLIP}_inpainting_results_sbs.mkv"
MP = f"outputs/fulldata_v2/clips/{CLIP}_origin/{CLIP}_inpainting_results_sbs.mp4"
SPL = f"video_data/splatting/{CLIP}_splatting_results.mp4"; TRN = f"video_data/train/{CLIP}_train.mp4"
FRAMES = [int(a) for a in sys.argv[1:]] or [30, 75, 120]

def rd(vr, idx): return torch.from_numpy(vr.get_batch(list(idx)).asnumpy()).permute(0, 3, 1, 2).float() / 255.0   # [T,3,H,W]
def crop_quadrants(fr):   # verbatim from minift/xcheck_mini_ft.py
    H, W = fr.shape[2] // 2, fr.shape[3] // 2
    TL, TR, BL, BR = fr[:, :, :H, :W], fr[:, :, :H, W:], fr[:, :, H:, :W], fr[:, :, H:, W:]
    h, w = H // 128 * 128, W // 128 * 128
    top, left = (h - 576) // 2, (w - 1024) // 2
    sl = (slice(None), slice(None), slice(top, top + 576), slice(left, left + 1024))
    return BR[:, :, :h, :w][sl], BL[:, :, :h, :w][sl].mean(dim=1, keepdim=True), TR[:, :, :h, :w][sl], TL[:, :, :h, :w][sl]
LAP = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]).view(1, 1, 3, 3)
def lap(x): return F.conv2d(x.mean(dim=1, keepdim=True), LAP)
def sharp(x): return (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean().item()

class Acc:   # streaming accumulator so the all-frame pass does not hold 2x151 float frames
    def __init__(s): s.n = 0; s.sum_d = 0.; s.sum_ad = 0.; s.sum_d2 = 0.; s.max_ad = 0.; s.lapA2 = 0.; s.lapB2 = 0.; s.lapD2 = 0.; s.lapA1 = 0.; s.lapB1 = 0.; s.shA = 0.; s.shB = 0.
    def add(s, A, B):
        d = B - A; k = A.shape[0]; s.n += k
        s.sum_d += d.mean().item() * k; s.sum_ad += d.abs().mean().item() * k; s.sum_d2 += d.pow(2).mean().item() * k; s.max_ad = max(s.max_ad, d.abs().max().item())
        la, lb, ld = lap(A), lap(B), lap(d)
        s.lapA2 += la.pow(2).mean().item() * k; s.lapB2 += lb.pow(2).mean().item() * k; s.lapD2 += ld.pow(2).mean().item() * k
        s.lapA1 += la.abs().mean().item() * k; s.lapB1 += lb.abs().mean().item() * k; s.shA += sharp(A) * k; s.shB += sharp(B) * k
    def report(s, label):
        n = s.n; mse = s.sum_d2 / n
        print(f"{label:28s} mean shift (mp4v - lossless) = {s.sum_d/n*255:+.4f}/255 | MAD = {s.sum_ad/n*255:.4f}/255 | maxabs = {s.max_ad*255:.0f}/255 | PSNR = {10*math.log10(1/max(mse,1e-18)):.2f} dB | "
              f"Laplacian energy ratio mp4v/lossless = {s.lapB2/s.lapA2:.4f} (mean|lap| ratio {s.lapB1/s.lapA1:.4f}; codec-error HF energy / target HF energy = {s.lapD2/s.lapA2:.4f}) | "
              f"sharp lossless {s.shA/n:.5f} mp4v {s.shB/n:.5f} (ratio {s.shB/s.shA:.4f})", flush=True)

def pair(A, B, label, dc=True):
    d = B - A; mse = d.pow(2).mean().item()
    print(f"{label:60s} mean shift = {d.mean().item()*255:+.4f}/255 | MAD = {d.abs().mean().item()*255:.4f}/255 | maxabs = {d.abs().max().item()*255:.0f}/255 | PSNR = {10*math.log10(1/max(mse,1e-18)):.2f} dB", flush=True)

vll, vmp, vspl, vtrn = VideoReader(LL, ctx=cpu(0)), VideoReader(MP, ctx=cpu(0)), VideoReader(SPL, ctx=cpu(0)), VideoReader(TRN, ctx=cpu(0))
assert len(vll) == len(vmp) == len(vspl) == len(vtrn) == 151, (len(vll), len(vmp), len(vspl), len(vtrn))
print(f"=== CONTROL A target readout, clip {CLIP}: lossless target {LL} vs old mp4v target {MP} ===")
print("--- RIGHT half (= the P1-null training target), per frame ---")
for f in FRAMES:
    A, B = rd(vll, [f]), rd(vmp, [f]); a = Acc(); a.add(A[:, :, :, 1024:], B[:, :, :, 1024:]); a.report(f"frame {f} right half")
print("--- RIGHT half, all 151 frames (streamed) ---")
acc_r, acc_l = Acc(), Acc()
for s0 in range(0, 151, 8):
    idx = range(s0, min(s0 + 8, 151)); A, B = rd(vll, idx), rd(vmp, idx)
    acc_r.add(A[:, :, :, 1024:], B[:, :, :, 1024:]); acc_l.add(A[:, :, :, :1024], B[:, :, :, :1024])
acc_r.report("all frames right half"); acc_l.report("all frames LEFT half")
print("--- LEFT-half registration and cond mismatch, per frame (crop_quadrants = the trainer's registered 576x1024 crop) ---")
for f in FRAMES:
    A, B = rd(vll, [f]), rd(vmp, [f]); Lll, Lmp = A[:, :, :, :1024], B[:, :, :, :1024]
    BRs, Ms, TRs, TLs = crop_quadrants(rd(vspl, [f])); BRt, Mt, TRt, TLt = crop_quadrants(rd(vtrn, [f]))
    pair(TLs, Lll, f"frame {f}: lossless sbs LEFT vs SPLATTING TL crop (must be 0)")
    pair(TLt, Lll, f"frame {f}: lossless sbs LEFT vs TRAIN-tile TL crop (trainer's mad_left)")
    pair(TLt, Lmp, f"frame {f}: mp4v sbs LEFT vs TRAIN-tile TL crop (old run saw 0.294)")
    pair(TLt, TLs, f"frame {f}: SPLATTING TL vs TRAIN TL (input DC offset)")
    pair(BRt, BRs, f"frame {f}: SPLATTING BR (pipeline cond) vs TRAIN BR (trainer cond)")
    pair(Mt, Ms, f"frame {f}: SPLATTING mask vs TRAIN mask")
    pair(TRt, A[:, :, :, 1024:], f"frame {f}: lossless target vs TRAIN TR (real GT, unregistered)")
    pair(BRt, A[:, :, :, 1024:], f"frame {f}: lossless target vs TRAIN BR (cond)")
print("TARGET_DIFF_DONE", flush=True)
