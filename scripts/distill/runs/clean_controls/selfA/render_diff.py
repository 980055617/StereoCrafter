"""Right-half pixel statistics of lossless renders against a reference render (CPU, streamed over all frames):
mean shift, MAD, PSNR, Laplacian energy ratio, the scorer's sharpness statistic, and the global mean level of each.
Used to check whether a trained checkpoint learned a DC shift or a high-frequency change relative to origin.
usage: python render_diff.py <ref_sbs.mkv> <label=path.mkv> [<label=path.mkv> ...]
"""
import os, sys, math, torch, torch.nn.functional as F
REPO = "/home/kawa/master_project/StereoCrafter"; os.chdir(REPO)
from decord import VideoReader, cpu
LAP = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]).view(1, 1, 3, 3)
def lap(x): return F.conv2d(x.mean(dim=1, keepdim=True), LAP)
def sharp(x): return (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean().item()
def rd(vr, idx): return torch.from_numpy(vr.get_batch(list(idx)).asnumpy()).permute(0, 3, 1, 2).float() / 255.0
ref = sys.argv[1]; vref = VideoReader(ref, ctx=cpu(0)); n = len(vref)
print(f"reference: {ref} ({n} frames)")
print(f"{'variant':34s} {'meanRef':>8s} {'meanVar':>8s} {'shift/255':>9s} {'MAD/255':>8s} {'PSNR':>7s} {'lapE var/ref':>12s} {'sharp ref':>9s} {'sharp var':>9s}")
for spec in sys.argv[2:]:
    label, path = spec.split("=", 1); v = VideoReader(path, ctx=cpu(0)); m = min(n, len(v))
    sA = sB = sd = sad = sd2 = lA = lB = shA = shB = 0.
    for s0 in range(0, m, 8):
        idx = range(s0, min(s0 + 8, m)); A = rd(vref, idx)[:, :, :, 1024:]; B = rd(v, idx)[:, :, :, 1024:]; k = A.shape[0]; d = B - A
        sA += A.mean().item() * k; sB += B.mean().item() * k; sd += d.mean().item() * k; sad += d.abs().mean().item() * k; sd2 += d.pow(2).mean().item() * k
        lA += lap(A).pow(2).mean().item() * k; lB += lap(B).pow(2).mean().item() * k; shA += sharp(A) * k; shB += sharp(B) * k
    print(f"{label[:34]:34s} {sA/m:8.5f} {sB/m:8.5f} {sd/m*255:+9.3f} {sad/m*255:8.3f} {10*math.log10(1/max(sd2/m,1e-18)):7.2f} {lB/lA:12.4f} {shA/m:9.5f} {shB/m:9.5f}")
print("RENDER_DIFF_DONE")
