"""CPU check: residual global horizontal disparity between the warped conditioning (BR) and the GT right eye (TR) inside the
576x1024 training/eval centre crop, frame 30, all train_gt28 clips + the 12 test clips.  MAD outside holes (BL<128), dx in [-64,64] step 2.
Also: true inter-eye disparity (TL vs TR) and how far the warp moved the left eye (BR vs TL)."""
import numpy as np, glob, os, statistics as st
from decord import VideoReader, cpu
def mad(a,b): return float(np.abs(a-b).mean())
def analyse(path):
    vr=VideoReader(path,ctx=cpu(0)); f=vr[min(30,len(vr)-1)].asnumpy().astype(np.float32)
    H,W=f.shape[0]//2,f.shape[1]//2; h,w=H//128*128,W//128*128
    TL=f[:H,:W][:h,:w]; TR=f[:H,W:2*W][:h,:w]; BR=f[H:2*H,W:2*W][:h,:w]; BL=f[H:2*H,:W][:h,:w].mean(2)
    top,left=(h-576)//2,(w-1024)//2; sl=(slice(top,top+576),slice(left,left+1024))
    TL,TR,BR,BL=TL[sl],TR[sl],BR[sl],BL[sl]; keep=(BL<128)
    def best(a,b,rng=64,step=2):
        res={}
        for dx in range(-rng,rng+1,step):
            aa=a[:, max(0,dx):1024+min(0,dx)]; bb=b[:, max(0,-dx):1024-max(0,dx)]; kk=keep[:, max(0,dx):1024+min(0,dx)]
            res[dx]=mad(aa[kk],bb[kk])
        d=min(res,key=res.get); return d,res[d],res[0]
    e=best(TL,TR); r=best(BR,TR); wl=best(BR,TL)
    return dict(eye_disp=e[0], eye_mad0=e[2], resid_disp=r[0], resid_mad0=r[2], resid_mad_best=r[1], warp_vs_left_mad0=wl[2], warp_moved=wl[0])
rows=[]
for p in sorted(glob.glob("video_data/train_gt28/*_train.mp4"))+[f"video_data/train/{c}_train.mp4" for c in "0042 0052 0125 0128 0141 0147 0170 0204 0225 0251 0259 0301".split()]:
    c=os.path.basename(p)[:4]; s="train" if "train_gt28" in p else "test"; r=analyse(p); rows.append((c,s,r)); print(c,s,r)
for s in ("train","test"):
    rs=[r for c,ss,r in rows if ss==s]
    print(s,"n=",len(rs),"median |cond->GT residual dx| px:",st.median(abs(r['resid_disp']) for r in rs),"frac>=8px:",sum(abs(r['resid_disp'])>=8 for r in rs)/len(rs),"median MAD(cond,GT)@0:",round(st.median(r['resid_mad0'] for r in rs),1),"median MAD(cond,left)@0:",round(st.median(r['warp_vs_left_mad0'] for r in rs),1))
