"""Distance between two right-eye outputs of the SAME input (no GT needed): mean per-frame LPIPS, PSNR, and tLP of each.
usage: python score_pair.py label a.mp4 b.mp4 [label a b ...]"""
import sys, torch, lpips, math
from decord import VideoReader, cpu
dev = "cuda"; net = lpips.LPIPS(net="alex").to(dev).eval()
def load(p):
    vr = VideoReader(p, ctx=cpu(0)); f = vr.get_batch(list(range(len(vr)))).asnumpy(); v = torch.from_numpy(f).permute(0, 3, 1, 2).float() / 255.; return v[:, :, :, v.shape[3] // 2:]
@torch.no_grad()
def lp(a, b, bs=16): return sum(net(a[i:i+bs].to(dev) * 2 - 1, b[i:i+bs].to(dev) * 2 - 1).sum().item() for i in range(0, len(a), bs)) / len(a)
@torch.no_grad()
def tlp(a, bs=16): return sum(net(a[i:i+bs+1].to(dev)[:-1] * 2 - 1, a[i:i+bs+1].to(dev)[1:] * 2 - 1).sum().item() for i in range(0, len(a) - 1, bs)) / (len(a) - 1)
print(f"{'pair':44s} {'n':>4s} {'LPIPS(a,b)':>10s} {'PSNR':>7s} {'tLP a':>7s} {'tLP b':>7s}")
args = sys.argv[1:]
for i in range(0, len(args), 3):
    lab, pa, pb = args[i:i+3]; a, b = load(pa), load(pb); n = min(len(a), len(b)); a, b = a[:n], b[:n]
    mse = (a - b).pow(2).mean().item(); print(f"{lab[:44]:44s} {n:4d} {lp(a, b):10.4f} {10*math.log10(1/max(mse,1e-12)):7.2f} {tlp(a):7.4f} {tlp(b):7.4f}", flush=True)
