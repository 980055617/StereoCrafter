"""Is the warped-vs-GT error outside the mask a global horizontal offset (disparity scale mismatch) or local? Also mask-value histogram.
usage: python shiftcheck.py clip ..."""
import sys, os, math, numpy as np, torch, torch.nn.functional as F
from decord import VideoReader, cpu
ROOT="/home/kawa/master_project/StereoCrafter"; TH,TW=576,1024; STEP=8
def load_u8(p): vr=VideoReader(p,ctx=cpu(0)); return vr.get_batch(list(range(0,len(vr),STEP))).asnumpy()
def to_t(a): return torch.from_numpy(np.ascontiguousarray(a)).permute(0,3,1,2).float()/255.
import json
for clip in sys.argv[1:]:
    J=json.load(open(f"{ROOT}/scripts/distill/runs/fulldata/beyond/crackcheck/{clip}_hotspots.json")); t0,l0,_=J["align"]; top,left=J["crop"]
    tile=load_u8(f"{ROOT}/video_data/train/{clip}_train.mp4"); H,W=tile.shape[1]//2,tile.shape[2]//2; G=to_t(tile[:,t0:t0+TH,W+l0:W+l0+TW]); GL=to_t(tile[:,t0:t0+TH,l0:l0+TW]); del tile
    sp=load_u8(f"{ROOT}/video_data/splatting/{clip}_splatting_results.mp4"); h,w=sp.shape[1]//2,sp.shape[2]//2
    mask=to_t(sp[:,h+top:h+top+TH,left:left+TW]).mean(1,keepdim=True); Wp=to_t(sp[:,h+top:h+top+TH,w+left:w+left+TW]); inL=to_t(sp[:,top:top+TH,left:left+TW]); del sp
    n=min(len(G),len(Wp)); G,GL,Wp,mask,inL=G[:n],GL[:n],Wp[:n],mask[:n],inL[:n]
    mb=(mask>0.5).float(); md=F.max_pool2d(mb,17,1,8); out=1-md
    edges=[0,0.005,0.02,0.05,0.1,0.25,0.5,0.75,0.99,1.01]; hist=torch.histogram(mask.flatten(),bins=torch.tensor(edges)).hist; hist=100*hist/hist.sum()
    print(f"[{clip}] mask value histogram %: "+" ".join(f"[{edges[i]},{edges[i+1]}):{hist[i]:.2f}" for i in range(len(edges)-1)))
    def err(a,b,wt): return ((a-b).abs().mean(1,keepdim=True)*wt).sum().item()/wt.sum().item()
    # global horizontal shift search of warped vs GT-right (outside mask)
    res=[]
    for dx in range(-48,49,2):
        if dx>=0: a=Wp[:,:,:,dx:]; b=G[:,:,:,:TW-dx]; o=out[:,:,:,dx:]
        else: a=Wp[:,:,:,:TW+dx]; b=G[:,:,:,-dx:]; o=out[:,:,:,:TW+dx]
        res.append((err(a,b,o),dx))
    res.sort(); print(f"[{clip}] warped-vs-GTright |err| outside mask: dx=0 -> {[r for r in res if r[1]==0][0][0]:.4f}; best global shift dx={res[0][1]} -> {res[0][0]:.4f}; 2nd {res[1]}")
    # how far apart are left and GT-right (true disparity magnitude) vs left and warped (synthetic disparity)?
    for name,(a,b) in {"GTleft vs GTright":(GL,G),"inL vs warped":(inL,Wp),"GTleft vs warped":(GL,Wp)}.items():
        rr=[]
        for dx in range(-64,65,2):
            if dx>=0: x=a[:,:,:,dx:]; y=b[:,:,:,:TW-dx]
            else: x=a[:,:,:,:TW+dx]; y=b[:,:,:,-dx:]
            rr.append(((x-y).abs().mean().item(),dx))
        rr.sort(); print(f"[{clip}] {name}: best global dx={rr[0][1]} |err|={rr[0][0]:.4f} (dx=0: {[r for r in rr if r[1]==0][0][0]:.4f})")
    # local: per-32px-block best shift (how non-uniform is the residual)
    bs=32; best=torch.zeros(n,TH//bs,TW//bs); e0=torch.zeros_like(best)
    shifts=list(range(-24,25,2)); E=[]
    for dx in shifts:
        if dx>=0: a=Wp[:,:,:,dx:]; b=G[:,:,:,:TW-dx]
        else: a=Wp[:,:,:,:TW+dx]; b=G[:,:,:,-dx:]
        e=(a-b).abs().mean(1,keepdim=True); e=F.pad(e,(max(dx,0),max(-dx,0)),value=1.0); E.append(F.avg_pool2d(e,bs)[:,0])
    E=torch.stack(E); ib=E.argmin(0); eb=E.min(0).values; e0=E[shifts.index(0)]
    sd=torch.tensor(shifts)[ib].float(); print(f"[{clip}] per-32px-block best dx: mean={sd.mean():.2f} std={sd.std():.2f} |dx|>=8 in {100*(sd.abs()>=8).float().mean():.1f}% blocks; block err at dx=0 {e0.mean():.4f} -> at best local dx {eb.mean():.4f}")
