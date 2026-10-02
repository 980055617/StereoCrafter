"""Verify the train tile layout: pairwise |err| between the 4 train quadrants and the 4 splat quadrants (frames 0, mid, last), plus a downsampled
tile PNG."""
import sys, numpy as np, torch
from decord import VideoReader, cpu
from PIL import Image
ROOT="/home/kawa/master_project/StereoCrafter"; OUT=f"{ROOT}/outputs/fulldata/beyond/crackcheck"
for clip in sys.argv[1:]:
    tr=VideoReader(f"{ROOT}/video_data/train/{clip}_train.mp4",ctx=cpu(0)); sp=VideoReader(f"{ROOT}/video_data/splatting/{clip}_splatting_results.mp4",ctx=cpu(0))
    for fi in (0,len(tr)//2,len(tr)-1):
        t=tr[fi].asnumpy().astype(np.float32)/255; s=sp[fi].asnumpy().astype(np.float32)/255; H,W=t.shape[0]//2,t.shape[1]//2
        q={"trTL":t[:H,:W],"trTR":t[:H,W:],"trBL":t[H:,:W],"trBR":t[H:,W:],"spTL":s[:H,:W],"spTR":s[:H,W:],"spBL":s[H:,:W],"spBR":s[H:,W:]}
        names=list(q); print(f"[{clip}] frame {fi}: "+" ".join(f"{a}-{b}={np.abs(q[a]-q[b]).mean():.4f}" for i,a in enumerate(names) for b in names[i+1:] if not (a.startswith('sp') and b.startswith('sp'))))
        # horizontal shift between trTL and trTR and between trTL and spBR (centre crop)
        def bestdx(a,b):
            r=[]
            for dx in range(-24,25,2):
                aa=a[:, max(dx,0):W+min(dx,0)]; bb=b[:, max(-dx,0):W+min(-dx,0)]; r.append((np.abs(aa-bb).mean(),dx))
            return min(r)
        print(f"[{clip}] frame {fi}: trTL vs trTR best dx={bestdx(q['trTL'],q['trTR'])}, trTL vs spBR best dx={bestdx(q['trTL'],q['spBR'])}, trTR vs spBR best dx={bestdx(q['trTR'],q['spBR'])}")
    im=Image.fromarray((tr[0].asnumpy())).resize((960,960*t.shape[0]//t.shape[1])); im2=Image.fromarray((sp[0].asnumpy())).resize((960,960*s.shape[0]//s.shape[1]))
    canvas=Image.new("RGB",(1930,im.height)); canvas.paste(im,(0,0)); canvas.paste(im2,(970,0)); p=f"{OUT}/{clip}_tiles_frame0.png"; canvas.save(p); print("wrote",p)
