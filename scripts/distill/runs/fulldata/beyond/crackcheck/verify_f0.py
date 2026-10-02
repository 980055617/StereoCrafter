"""Independent re-check of Report C F0: is train.mp4 top-right the left eye? CPU only."""
import sys, numpy as np
from decord import VideoReader, cpu
def q(v, idx):
    f = v.get_batch([idx]).asnumpy()[0].astype(np.float32)/255.; H,W = f.shape[0]//2, f.shape[1]//2
    return {'TL':f[:H,:W],'TR':f[:H,W:2*W],'BL':f[H:2*H,:W],'BR':f[H:2*H,W:2*W]}
def mad(a,b): return float(np.abs(a-b).mean())
def best_dx(a,b,rng=24):
    best=(1e9,0)
    for dx in range(-rng,rng+1):
        if dx>=0: e=mad(a[:,dx:],b[:,:a.shape[1]-dx])
        else: e=mad(a[:,:dx],b[:,-dx:])
        if e<best[0]: best=(e,dx)
    return best
for clip in sys.argv[1:]:
    tr=VideoReader(f"video_data/train/{clip}_train.mp4",ctx=cpu(0)); sp=VideoReader(f"video_data/splatting/{clip}_splatting_results.mp4",ctx=cpu(0))
    n=min(len(tr),len(sp))
    for idx in [0, n//2, n-1]:
        t=q(tr,idx); s=q(sp,idx)
        # crop a center region to avoid the last-row/border effects
        c=lambda x: x[100:-100,100:-100]
        print(f"[{clip}] f{idx} n={n} tr{tr[0].shape} sp{sp[0].shape} | trTR-spTL={mad(c(t['TR']),c(s['TL'])):.4f} trTL-spTL={mad(c(t['TL']),c(s['TL'])):.4f} trTL-trTR={mad(c(t['TL']),c(t['TR'])):.4f} trBR-spBR={mad(c(t['BR']),c(s['BR'])):.4f} trBL-spBL={mad(c(t['BL']),c(s['BL'])):.4f}"
              f" | bestdx warped(spBR)->trTR={best_dx(c(s['BR']),c(t['TR']))} warped->trTL={best_dx(c(s['BR']),c(t['TL']))} trTL->trTR={best_dx(c(t['TL']),c(t['TR']),12)}", flush=True)
