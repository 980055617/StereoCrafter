"""Self-aligning evaluation of an inpainting SBS output against a 2x2 train tile.

Why this exists: `utils/inpainting.py::read_and_prepare_video` crops the input
to multiples of 128 from the TOP-LEFT (1080 -> 1024, dropping the bottom 56
rows) *before* the target_height/target_width center crop. So a 576x1024
output from `inpainting_inference.py` is taken at source rows 224..800, while
`scripts/evaluate_inpainting_train_tile.py` center-crops the GT from the full
1080 (rows 252..828) -- a 28px vertical misalignment that silently depresses
every PSNR by 1-3dB, and penalizes SHARP outputs far more than blurry ones
(a blurry frame barely notices a 28px shift), inverting model rankings.

This script discovers the true offset empirically from the LEFT eye, which is
passed through untouched and therefore must match the GT almost exactly (40dB+
when aligned, ~12dB when not), then scores the right eye at that alignment.
Use this instead of the fixed center-crop evaluator whenever comparing outputs
produced by different inference scripts."""
import sys, torch
from decord import VideoReader, cpu

def load(p, step=1):
    vr=VideoReader(p,ctx=cpu(0)); f=vr.get_batch(list(range(0,len(vr),step))).asnumpy()
    return torch.from_numpy(f).permute(0,3,1,2).float()/255.

def psnr(mse): return 10*torch.log10(torch.tensor(1.0/max(mse,1e-12))).item()

def main(gen_path, train_tile="video_data/train/0160_train.mp4"):
    train=load(train_tile); H,W=train.shape[2]//2,train.shape[3]//2
    gtL=train[:,:,:H,:W]; gtR=train[:,:,:H,W:2*W]
    mskF=train[:,:,H:2*H,:W].mean(dim=1,keepdim=True); wrpF=train[:,:,H:2*H,W:2*W]
    v=load(gen_path); half=v.shape[3]//2
    L=v[:,:,:,:half]; R=v[:,:,:,half:]
    n=min(len(L),len(gtL)); L,R=L[:n],R[:n]
    h,w=L.shape[2],L.shape[3]
    # coarse->fine alignment search on the pass-through left eye
    best=((0,0),-1)
    for step,rng in ((4,52),(1,6)):
        cy,cx=best[0]
        for dy in range(cy-rng,cy+rng+1,step):
            for dx in range(cx-rng,cx+rng+1,step):
                t0=(H-h)//2+dy; l0=(W-w)//2+dx
                if t0<0 or l0<0 or t0+h>H or l0+w>W: continue
                m=(L[::10]-gtL[:n:10,:,t0:t0+h,l0:l0+w]).pow(2).mean().item()
                if psnr(m)>best[1]: best=((dy,dx),psnr(m))
    (dy,dx),align_psnr=best
    t0=(H-h)//2+dy; l0=(W-w)//2+dx
    gr=gtR[:n,:,t0:t0+h,l0:l0+w]; mk=mskF[:n,:,t0:t0+h,l0:l0+w]; wp=wrpF[:n,:,t0:t0+h,l0:l0+w]
    mb=(mk>0.5).float(); inv=1.0-mb
    def reg(pred,m):
        if m is None: return psnr((pred-gr).pow(2).mean().item())
        mr=m.expand_as(pred); d=float(mr.sum())
        return psnr(float(((pred-gr).pow(2)*mr).sum())/d) if d>0 else float('nan')
    print(f"{gen_path}")
    print(f"  align offset (dy,dx)=({dy},{dx})  left-eye PSNR={align_psnr:.2f}  frames={n}")
    print(f"  mask_gen={reg(R,mb):7.3f}  mask_warp={reg(wp,mb):7.3f}   "
          f"inv_gen={reg(R,inv):7.3f}  inv_warp={reg(wp,inv):7.3f}   all_gen={reg(R,None):7.3f}")

if __name__=="__main__":
    for p in sys.argv[1:]: main(p)
