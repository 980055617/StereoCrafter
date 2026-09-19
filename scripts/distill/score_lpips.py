import sys, torch, lpips, math
from decord import VideoReader, cpu
def load(p,step=4):
    vr=VideoReader(p,ctx=cpu(0)); f=vr.get_batch(list(range(0,len(vr),step))).asnumpy()
    return torch.from_numpy(f).permute(0,3,1,2).float()/255.
tile=load("video_data/train/0160_train.mp4")
gtR=tile[:,:,224:800,1920+448:1920+1472]
mask=tile[:,:,1080+224:1080+800,448:1472].mean(dim=1,keepdim=True)
mb=(mask>0.5).float()
net=lpips.LPIPS(net='alex').cuda().eval()
rows=[]
for p in sys.argv[1:]:
    v=load(p); g=v[:,:,:,v.shape[3]//2:]
    n=min(len(g),len(gtR)); g,t,m=g[:n],gtR[:n],mb[:n]
    mr=m.expand_as(g); mse_m=float(((g-t).pow(2)*mr).sum())/float(mr.sum())
    tot=0.
    with torch.no_grad():
        for i in range(0,n,4):
            tot+=float(net((g[i:i+4].cuda()*2-1),(t[i:i+4].cuda()*2-1)).sum())
    sh=(g[:,:,:,1:]-g[:,:,:,:-1]).abs().mean().item()
    rows.append((p.split('/')[-2], tot/n, 10*math.log10(1/mse_m), sh))
rows.sort(key=lambda r:r[1])
print(f"{'config':52s} {'LPIPS':>8s} {'maskPSNR':>9s} {'sharp':>8s}")
gs=(gtR[:,:,:,1:]-gtR[:,:,:,:-1]).abs().mean().item()
print(f"{'--- GT reference ---':52s} {0.0:8.4f} {'-':>9s} {gs:8.4f}")
for r in rows: print(f"{r[0][:52]:52s} {r[1]:8.4f} {r[2]:9.3f} {r[3]:8.4f}")
