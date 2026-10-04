"""CPU identity test: tracked utils.inpainting.read_and_prepare_video vs lowmem_reader, on the first K frames
(K=20 -> two lowmem chunks), then inpainting_inference._center_crop_frames at each test resolution.
Requires torch.equal (bitwise values) AND identical strides for left / warped / mask, plus identical fps and the
strides of the per-window .clone() / slice the inference loop takes.  Prints IDENTITY_PASS or IDENTITY_FAIL."""
import os, sys
REPO = "/home/kawa/master_project/StereoCrafter"; sys.path.insert(0, REPO); os.chdir(REPO)
sys.path.insert(0, f"{REPO}/scripts/distill/runs/finalcheck_20261004/validate")
import torch, decord
import utils.inpainting as UI
from inpainting_inference import _center_crop_frames
from lowmem_reader import read_and_prepare_video_lowmem
K = int(os.environ.get("K", "20"))
class LimitedVR:
    def __init__(self, path, ctx=None):
        self.vr = decord.VideoReader(path, ctx=ctx if ctx is not None else decord.cpu(0))
    def __len__(self): return min(K, len(self.vr))
    def __getitem__(self, i): return self.vr[i]
    def get_batch(self, idx):
        assert max(idx) < K
        return self.vr.get_batch(idx)
    def get_avg_fps(self): return self.vr.get_avg_fps()
UI.VideoReader = LimitedVR                     # the tracked reader resolves VideoReader from its module globals
ok = True
for clip in sys.argv[1:]:
    p = f"video_data/splatting/{clip}_splatting_results.mp4"
    fa, la, wa, ma = UI.read_and_prepare_video(p)
    fb, lb, wb, mb = read_and_prepare_video_lowmem(p, _reader=LimitedVR)
    same_full = all(torch.equal(x, y) and x.stride() == y.stride() and x.shape == y.shape for x, y in ((la, lb), (wa, wb), (ma, mb)))
    print(f"{clip}: fps {fa} vs {fb}; full shapes {tuple(la.shape)} {tuple(ma.shape)}; strides L {la.stride()} vs {lb.stride()}, "
          f"M {ma.stride()} vs {mb.stride()}; full-size identical={same_full}", flush=True)
    ok &= (fa == fb) and same_full
    for (h, w) in ((576, 1024), (1024, 1792), (1024, 1920)):
        r = []
        for x, y in ((la, lb), (wa, wb), (ma, mb)):
            cx, cy = _center_crop_frames(x, h, w), _center_crop_frames(y, h, w)
            e = torch.equal(cx, cy) and cx.stride() == cy.stride() and cx.storage_offset() == cy.storage_offset()
            e &= cx[0:14].clone().stride() == cy[0:14].clone().stride() and cx[3:17].stride() == cy[3:17].stride()
            r.append(e)
        ok &= all(r)
        print(f"  {clip} {h}x{w}: left {r[0]} warped {r[1]} mask {r[2]}", flush=True)
    del la, wa, ma, lb, wb, mb
print("IDENTITY_PASS" if ok else "IDENTITY_FAIL", flush=True)
