# CPU COPY for deep_20261004/external_models (PREREG_r2.txt, deviation D3): identical to
# scripts/distill/runs/fulldata_v2/beyond4/score_clip_ll.py except every .cuda() -> .to(DEV) with DEV='cpu'
# (the GPU-0 lock had 9+ waiters).  Gate: origin_ll re-scored here must reproduce its published lpips.
# Copy of scripts/distill/score_clip.py with (a) container-agnostic paths (decord reads FFV1 .mkv
# bit-exactly) and (b) an extra machine-readable "ROW" line carrying the GT sharpness and rightPSNR.
# The alignment search, the LPIPS accumulation and the sharpness statistic are UNCHANGED.
import sys, os, torch, lpips, math
DEV='cpu'; torch.set_num_threads(int(os.environ.get('NTHREADS','16')))
STEP=int(os.environ.get("SCORE_STEP","4"))
from decord import VideoReader, cpu
def load(p,step=None):
    step=STEP if step is None else step
    vr=VideoReader(p,ctx=cpu(0)); f=vr.get_batch(list(range(0,len(vr),step))).asnumpy()
    return torch.from_numpy(f).permute(0,3,1,2).float()/255.
net=lpips.LPIPS(net='alex').to(DEV).eval()
print(f"{'clip/config':34s} {'offset':>9s} {'leftPSNR':>9s} {'LPIPS':>8s} {'sharp':>8s} {'rPSNR':>8s} {'nf':>4s}")
cur=None; tile=None
for spec in sys.argv[1:]:
    clip,path = spec.split('=',1)
    if clip!=cur:
        tile=load(f"video_data/train/{clip}_train.mp4"); cur=clip
        H,W=tile.shape[2]//2,tile.shape[3]//2
        gtL=tile[:,:,:H,:W]; gtR=tile[:,:,:H,W:2*W]
        gs=(gtR[:,:,:,1:]-gtR[:,:,:,:-1]).abs().mean().item()
        print(f"{f'--- {clip} GT ---':34s} {'':>9s} {'':>9s} {0.0:8.4f} {gs:8.4f}")
    v=load(path); half=v.shape[3]//2
    L=v[:,:,:,:half]; R=v[:,:,:,half:]
    n=min(len(L),len(gtL)); L,R=L[:n],R[:n]; h,w=L.shape[2],L.shape[3]
    best=((0,0),-1)
    for st,rng in ((4,60),(1,6)):
        cy,cx=best[0]
        for dy in range(cy-rng,cy+rng+1,st):
            for dx in range(cx-rng,cx+rng+1,st):
                t0=(H-h)//2+dy; l0=(W-w)//2+dx
                if t0<0 or l0<0 or t0+h>H or l0+w>W: continue
                m=(L[::6]-gtL[:n:6,:,t0:t0+h,l0:l0+w]).pow(2).mean().item()
                ps=10*math.log10(1/max(m,1e-12))
                if ps>best[1]: best=((dy,dx),ps)
    (dy,dx),al=best; t0=(H-h)//2+dy; l0=(W-w)//2+dx
    t=gtR[:n,:,t0:t0+h,l0:l0+w]; tot=0.
    with torch.no_grad():
        for i in range(0,n,4): tot+=float(net((R[i:i+4].to(DEV)*2-1),(t[i:i+4].to(DEV)*2-1)).sum())
    sh=(R[:,:,:,1:]-R[:,:,:,:-1]).abs().mean().item()
    rmse=(R-t).pow(2).mean().item(); rp=10*math.log10(1/max(rmse,1e-12))
    tag=path.split('/')[-2]
    print(f"{tag[:34]:34s} {f'({dy},{dx})':>9s} {al:9.2f} {tot/n:8.4f} {sh:8.4f} {rp:8.3f} {n:4d}")
    print(f"ROW clip={clip} tag={tag} dy={dy} dx={dx} leftPSNR={al:.4f} lpips={tot/n:.6f} sharp={sh:.6f} gtSharp={gs:.6f} rightPSNR={rp:.4f} n={n} path={path}")
