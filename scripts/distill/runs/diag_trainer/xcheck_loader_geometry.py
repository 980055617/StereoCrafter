"""CPU check of utils/training_batches.py:71-72,97-100 quadrant split (128-rounded tile) vs the true-half split of utils/inpainting.py:147-158,
plus the overlap-teacher adjacency rate under the 24-window subsample of inpainting_train.py:3640-3642."""
import numpy as np, random
from decord import VideoReader, cpu
def mad(a,b): return float(np.abs(a-b).mean())
for clip in ["0154","0011","0358"]:
    vr=VideoReader(f"video_data/train_gt28/{clip}_train.mp4",ctx=cpu(0)); f=vr[30].asnumpy().astype(np.float32)
    H,W=f.shape[0]//2,f.shape[1]//2; th,tw=(H//128)*128,(W//128)*128; dh,dw=H-th,W-tw
    tTR=f[:H,W:2*W]; tBR=f[H:2*H,W:2*W]; tTL=f[:H,:W]; fr=f[:2*th,:2*tw]; lTR=fr[:th,tw:]; lBL=fr[th:,:tw]; lBR=fr[th:,tw:]
    print(clip, f.shape[:2], "tile",(th,tw),"offset (dh,dw)=",(dh,dw))
    print(f"  loader-target vs trueTR col-shifted by dw: {mad(lTR[:, dw:], tTR[:th, :tw-dw]):.2f} | unshifted {mad(lTR, tTR[:th,:tw]):.2f}")
    print(f"  loader-cond vs trueBR shifted (dh,dw): {mad(lBR[dh:, dw:], tBR[:th-dh, :tw-dw]):.2f} | unshifted {mad(lBR, tBR[:th,:tw]):.2f}")
    print(f"  loader-mask top {dh} rows vs left-eye bottom rows: {mad(lBL[:dh], tTL[H-dh:H, :tw]) if dh>0 else float('nan'):.2f}  -> relative target-vs-cond displacement {dh} rows = {dh/8:.1f} latent rows")
tot=adj=0
for epoch in (1,2):
    for video_idx in range(1,17):
        sel=sorted(random.Random(7*1000003+epoch*7919+video_idx).sample(range(149),24)); tot+=len(sel); adj+=sum(1 for a,b in zip(sel,sel[1:]) if b==a+1)
print(f"overlap-teacher: adjacent-window fraction {adj/tot:.3f} x 0.3 => {0.3*adj/tot*100:.1f}% of windows teacher-forced")
