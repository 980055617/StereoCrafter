"""GT-free sharpness: mean |Laplacian| (gray) in the NON-mask region (hard mask dilated 8 px excluded) for
left eye (the source), warped input, origin output. Deployed 576x1024 crop as score_composite.py; every 8th frame. CPU only."""
import sys, numpy as np, torch, torch.nn.functional as F
from decord import VideoReader, cpu
TH,TW=576,1024
def load(p,step=8):
    vr=VideoReader(p,ctx=cpu(0)); f=vr.get_batch(list(range(0,len(vr),step))).asnumpy(); return torch.from_numpy(f).permute(0,3,1,2).float()/255.
lap=torch.tensor([[0,1,0],[1,-4,1],[0,1,0]],dtype=torch.float32).view(1,1,3,3)
def sharp(x,keep):  # x: N,3,H,W ; keep: N,1,H,W
    g=x.mean(1,keepdim=True); l=F.conv2d(g,lap,padding=1).abs(); k=keep[:,:,1:-1,1:-1]; return float((l[:,:,1:-1,1:-1]*k).sum()/k.sum())
rows=[]
for clip in sys.argv[1:]:
    sp=load(f"video_data/splatting/{clip}_splatting_results.mp4"); h,w=sp.shape[2]//2,sp.shape[3]//2; h128,w128=h//128*128,w//128*128; top,left=(h128-TH)//2,(w128-TW)//2
    L=sp[:,:,top:top+TH,left:left+TW]; mask=sp[:,0:1,h+top:h+top+TH,left:left+TW]; W_=sp[:,:,h+top:h+top+TH,w+left:w+left+TW]
    o=load(f"outputs/fulldata/clips/{clip}_origin/{clip}_inpainting_results_sbs.mp4"); O=o[:,:,:,o.shape[3]//2:]
    n=min(len(L),len(O)); L,mask,W_,O=L[:n],mask[:n],W_[:n],O[:n]
    m=(mask>0.5).float(); m=F.max_pool2d(m,17,1,8); keep=1-m
    sL,sW,sO=sharp(L,keep),sharp(W_,keep),sharp(O,keep)
    rows.append((clip,sL,sW,sO)); print(f"{clip} frames={n} lap_left={sL:.4f} lap_warped={sW:.4f} lap_origin={sO:.4f}  warped/left={sW/sL:.3f} origin/left={sO/sL:.3f} origin/warped={sO/sW:.3f}",flush=True)
a=np.array([r[1:] for r in rows]); print(f"MEAN{len(rows)} lap_left={a[:,0].mean():.4f} lap_warped={a[:,1].mean():.4f} lap_origin={a[:,2].mean():.4f} warped/left={np.mean(a[:,1]/a[:,0]):.3f} origin/left={np.mean(a[:,2]/a[:,0]):.3f} origin/warped={np.mean(a[:,2]/a[:,1]):.3f}")
