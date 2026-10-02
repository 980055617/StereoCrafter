"""Does compositing the known (non-mask) pixels back over the generated right eye cut the gap to GT?
For each clip: LPIPS vs GT of (a) the raw generated right eye, (b) composite = warped where mask==0, generated where mask==1
(mask dilated by D px, feathered by F px). usage: python score_composite.py clip=out.mp4 [...]   env: DIL (default 8), FEATHER (8)"""
import sys, os, math, torch, torch.nn.functional as F, lpips
from decord import VideoReader, cpu
DIL=int(os.environ.get("DIL","8")); FE=int(os.environ.get("FEATHER","8")); dev="cuda"; net=lpips.LPIPS(net="alex").to(dev).eval(); TH,TW=576,1024
def load(p, step=4):
    vr=VideoReader(p,ctx=cpu(0)); f=vr.get_batch(list(range(0,len(vr),step))).asnumpy(); return torch.from_numpy(f).permute(0,3,1,2).float()/255.
def align(L,gtL,H,W):
    n=min(len(L),len(gtL)); h,w=L.shape[2],L.shape[3]; best=((0,0),-1)
    for st,rng in ((4,60),(1,6)):
        cy,cx=best[0]
        for dy in range(cy-rng,cy+rng+1,st):
            for dx in range(cx-rng,cx+rng+1,st):
                t0=(H-h)//2+dy; l0=(W-w)//2+dx
                if t0<0 or l0<0 or t0+h>H or l0+w>W: continue
                m=(L[:n:6]-gtL[:n:6,:,t0:t0+h,l0:l0+w]).pow(2).mean().item(); ps=10*math.log10(1/max(m,1e-12))
                if ps>best[1]: best=((dy,dx),ps)
    (dy,dx),_=best; return (H-h)//2+dy,(W-w)//2+dx
@torch.no_grad()
def lp(a,b,bs=8): return sum(net(a[i:i+bs].to(dev)*2-1,b[i:i+bs].to(dev)*2-1).sum().item() for i in range(0,len(a),bs))/len(a)
print(f"{'clip/config':30s} {'raw':>8s} {'composite':>10s} {'mask%':>6s} {'warped-only':>11s} {'maskPSNR':>9s} {'maskPSNR_warped':>16s}")
cur=None
for spec in sys.argv[1:]:
    clip,path=spec.split("=",1)
    if clip!=cur:
        tile=load(f"video_data/train/{clip}_train.mp4"); H,W=tile.shape[2]//2,tile.shape[3]//2; gtL,gtR=tile[:,:,:H,:W],tile[:,:,:H,W:2*W]
        sp=load(f"video_data/splatting/{clip}_splatting_results.mp4"); h,w=sp.shape[2]//2,sp.shape[3]//2; h128,w128=h//128*128,w//128*128; top,left=(h128-TH)//2,(w128-TW)//2
        mask=sp[:,0:1,h+top:h+top+TH,left:left+TW]; warped=sp[:,:,h+top:h+top+TH,w+left:w+left+TW]; cur=clip
        m=(mask>0.5).float(); k=2*DIL+1; m=F.max_pool2d(m,k,1,k//2) if DIL>0 else m
        if FE>0: m=F.avg_pool2d(m,2*FE+1,1,FE)
    v=load(path); half=v.shape[3]//2; L,R=v[:,:,:,:half],v[:,:,:,half:]; n=min(len(R),len(gtR),len(m)); R=R[:n]; t0,l0=align(L[:n],gtL,H,W); G=gtR[:n,:,t0:t0+TH,l0:l0+TW]
    comp=warped[:n]*(1-m[:n])+R*m[:n]
    mb=(mask[:n]>0.5).float()
    def mpsnr(x):
        e=((x-G)**2*mb).sum()/(mb.sum()*3+1e-9); return 10*math.log10(1/max(e.item(),1e-12))
    tag=path.split("/")[-2]; print(f"{tag[:30]:30s} {lp(R,G):8.4f} {lp(comp,G):10.4f} {100*mb.mean():6.2f} {lp(warped[:n],G):11.4f} {mpsnr(R):9.2f} {mpsnr(warped[:n]):16.2f}", flush=True)
