import sys, torch, lpips, math
from decord import VideoReader, cpu
def load(p,step=4):
    vr=VideoReader(p,ctx=cpu(0)); f=vr.get_batch(list(range(0,len(vr),step))).asnumpy()
    return torch.from_numpy(f).permute(0,3,1,2).float()/255.
tile=load("video_data/train/0160_train.mp4")
H,W=1080,1920
gtL_full=tile[:,:,:H,:W]; gtR_full=tile[:,:,:H,W:2*W]
net=lpips.LPIPS(net='alex').cuda().eval()
print(f"{'config':40s} {'res':>10s} {'offset':>9s} {'leftPSNR':>9s} {'LPIPS':>8s} {'sharp':>8s}")
gs=(gtR_full[:,:,:,1:]-gtR_full[:,:,:,:-1]).abs().mean().item()
print(f"{'--- GT ---':40s} {'':>10s} {'':>9s} {'':>9s} {0.0:8.4f} {gs:8.4f}")
for p in sys.argv[1:]:
    v=load(p); half=v.shape[3]//2
    L=v[:,:,:,:half]; R=v[:,:,:,half:]
    n=min(len(L),len(gtL_full)); L,R=L[:n],R[:n]
    h,w=L.shape[2],L.shape[3]
    best=((0,0),-1)
    for step,rng in ((4,60),(1,6)):
        cy,cx=best[0]
        for dy in range(cy-rng,cy+rng+1,step):
            for dx in range(cx-rng,cx+rng+1,step):
                t0=(H-h)//2+dy; l0=(W-w)//2+dx
                if t0<0 or l0<0 or t0+h>H or l0+w>W: continue
                m=(L[::6]-gtL_full[:n:6,:,t0:t0+h,l0:l0+w]).pow(2).mean().item()
                ps=10*math.log10(1/max(m,1e-12))
                if ps>best[1]: best=((dy,dx),ps)
    (dy,dx),lp_align=best
    t0=(H-h)//2+dy; l0=(W-w)//2+dx
    t=gtR_full[:n,:,t0:t0+h,l0:l0+w]
    tot=0.
    with torch.no_grad():
        for i in range(0,n,2):
            tot+=float(net((R[i:i+2].cuda()*2-1),(t[i:i+2].cuda()*2-1)).sum())
    sh=(R[:,:,:,1:]-R[:,:,:,:-1]).abs().mean().item()
    print(f"{p.split('/')[-2][:40]:40s} {f'{h}x{w}':>10s} {f'({dy},{dx})':>9s} {lp_align:9.2f} {tot/n:8.4f} {sh:8.4f}")
