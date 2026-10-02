"""TASK C crack-check: find the largest warped-vs-GT error hotspots OUTSIDE the (8px-dilated) splatting mask and cut
GT | warped | origin | composite crops for visual inspection.  Crop chain + alignment copied from scripts/distill/score_composite.py.
usage: python hotspots.py 0160 0042 ...   env: STEP (frame stride, 4), CROP (256), DIL (8), FEATHER (8), SEED (0)"""
import sys, os, math, json, time, numpy as np, torch, torch.nn.functional as F
from decord import VideoReader, cpu
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
torch.set_num_threads(8)
ROOT="/home/kawa/master_project/StereoCrafter"; OUT=f"{ROOT}/outputs/fulldata/beyond/crackcheck"; RUN=f"{ROOT}/scripts/distill/runs/fulldata/beyond/crackcheck"
TH,TW=576,1024; STEP=int(os.environ.get("STEP","4")); CROP=int(os.environ.get("CROP","256")); DIL=int(os.environ.get("DIL","8")); FE=int(os.environ.get("FEATHER","8")); SEED=int(os.environ.get("SEED","0"))
NHOT=3; NCTRL=2; STRIDE=16
FONT=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",13); FONTB=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",14)

def load_u8(p):
    vr=VideoReader(p,ctx=cpu(0)); idx=list(range(0,len(vr),STEP)); return vr.get_batch(idx).asnumpy(), idx
def to_t(a): return torch.from_numpy(np.ascontiguousarray(a)).permute(0,3,1,2).float()/255.
def psnr(a,b): m=(a-b).pow(2).mean().item(); return 10*math.log10(1/max(m,1e-12))
def align(L,gtL_u8,H,W):
    # identical search to score_composite.align, but gtL kept uint8 and only the [::6] subset converted
    n=min(len(L),len(gtL_u8)); h,w=L.shape[2],L.shape[3]; Ls=L[:n:6]; gs=to_t(gtL_u8[:n:6]); best=((0,0),-1)
    for st,rng in ((4,60),(1,6)):
        cy,cx=best[0]
        for dy in range(cy-rng,cy+rng+1,st):
            for dx in range(cx-rng,cx+rng+1,st):
                t0=(H-h)//2+dy; l0=(W-w)//2+dx
                if t0<0 or l0<0 or t0+h>H or l0+w>W: continue
                m=(Ls-gs[:,:,t0:t0+h,l0:l0+w]).pow(2).mean().item(); ps=10*math.log10(1/max(m,1e-12))
                if ps>best[1]: best=((dy,dx),ps)
    (dy,dx),ps=best; return (H-h)//2+dy,(W-w)//2+dx,ps

def u8(t): return (t.clamp(0,1)*255).round().byte().permute(1,2,0).numpy()
def label_img(im,txt,color=(255,255,0)):
    d=ImageDraw.Draw(im); d.rectangle([0,0,im.width,18],fill=(0,0,0)); d.text((3,2),txt,font=FONT,fill=color); return im

def run(clip):
    t=time.time(); log=[]; P=lambda *a: (print(*a,flush=True), log.append(" ".join(str(x) for x in a)))
    # ---- GT tile
    tile,idx=load_u8(f"{ROOT}/video_data/train/{clip}_train.mp4"); H,W=tile.shape[1]//2,tile.shape[2]//2
    gtL_u8=tile[:,:H,:W]; gtR_u8=tile[:,:H,W:2*W]; del tile
    # ---- splatting tile -> deployed crop chain (score_composite.py lines 23-24)
    sp,_=load_u8(f"{ROOT}/video_data/splatting/{clip}_splatting_results.mp4"); h,w=sp.shape[1]//2,sp.shape[2]//2; h128,w128=h//128*128,w//128*128; top,left=(h128-TH)//2,(w128-TW)//2
    inL=to_t(sp[:,top:top+TH,left:left+TW]); mask_raw=to_t(sp[:,h+top:h+top+TH,left:left+TW]).mean(1,keepdim=True); warped=to_t(sp[:,h+top:h+top+TH,w+left:w+left+TW]); del sp
    # ---- origin output
    v,_=load_u8(f"{ROOT}/outputs/fulldata/clips/{clip}_origin/{clip}_inpainting_results_sbs.mp4"); v=to_t(v); half=v.shape[3]//2; L,R=v[:,:,:,:half],v[:,:,:,half:]
    n=min(len(R),len(warped),len(gtR_u8)); L,R,warped,mask_raw,inL=L[:n],R[:n],warped[:n],mask_raw[:n],inL[:n]
    t0,l0,alps=align(L,gtL_u8,H,W); G=to_t(gtR_u8[:n,t0:t0+TH,l0:l0+TW]); GL=to_t(gtL_u8[:n,t0:t0+TH,l0:l0+TW]); del gtL_u8,gtR_u8
    P(f"[{clip}] n={n} frames(step {STEP}) quadrant {H}x{W} splat-crop top,left=({top},{left}) gt-align t0,l0=({t0},{l0}) alignPSNR(originL vs gtL)={alps:.2f} | PSNR(originL vs splat-inL)={psnr(L,inL):.2f} PSNR(splat-inL vs gtL)={psnr(inL,GL):.2f}")
    # ---- masks
    mb=(mask_raw>0.5).float(); k=2*DIL+1; md=F.max_pool2d(mb,k,1,k//2); m=F.avg_pool2d(md,2*FE+1,1,FE) if FE>0 else md
    comp=warped*(1-m)+R*m
    soft=((mask_raw>0.02)&(mask_raw<=0.5)).float()   # partially-covered splat pixels that the 0.5 threshold calls "known"
    # ---- error maps (channel-mean abs error), outside = 1-md
    eW=(warped-G).abs().mean(1,keepdim=True); eO=(R-G).abs().mean(1,keepdim=True); eC=(comp-G).abs().mean(1,keepdim=True); out=1-md
    def stat(e,w): return (e*w).sum().item()/max(w.sum().item(),1)
    P(f"[{clip}] mask%: hard={100*mb.mean():.2f} dilated8={100*md.mean():.2f} soft(0.02<m<=0.5)={100*soft.mean():.3f}")
    P(f"[{clip}] mean|err| outside dilated mask: warped={stat(eW,out):.4f} origin={stat(eO,out):.4f} composite={stat(eC,out):.4f} | inside hard mask: warped={stat(eW,mb):.4f} origin={stat(eO,mb):.4f}")
    for th in (0.1,0.25,0.5): P(f"[{clip}] frac outside-mask px with |err|>{th}: warped={100*stat((eW>th).float(),out):.3f}% origin={100*stat((eO>th).float(),out):.3f}%")
    P(f"[{clip}] soft-mask px (0.02<m<=0.5, outside dilated hard mask): count%={100*stat(soft,out)*out.mean():.3f} mean|err| warped={stat(eW,soft*out):.4f} origin={stat(eO,soft*out):.4f}  vs mask==0 px: warped={stat(eW,(mask_raw<=0.02).float()*out):.4f}")
    # ---- error vs distance-to-hard-mask (frames with any mask)
    dist=torch.full_like(mb,1e4)
    for i in range(n):
        if mb[i,0].sum()>0: dist[i,0]=torch.from_numpy(ndimage.distance_transform_edt(1-mb[i,0].numpy()).astype(np.float32))
    bins=[(0,8),(8,16),(16,32),(32,64),(64,128),(128,256),(256,1e5)]
    P(f"[{clip}] |err| vs distance-to-hard-mask (px): "+" ".join(f"[{a}-{b if b<1e5 else 'inf'}): W={stat(eW,((dist>=a)&(dist<b)).float()):.4f}/O={stat(eO,((dist>=a)&(dist<b)).float()):.4f}/n={100*((dist>=a)&(dist<b)).float().mean():.2f}%" for a,b in bins))
    # ---- hotspot search: CROPxCROP window mean of eW over outside pixels, stride STRIDE, require >=50% outside px
    ws=F.avg_pool2d(eW*out,CROP,STRIDE); wc=F.avg_pool2d(out,CROP,STRIDE); score=torch.where(wc>=0.5,ws/wc.clamp(min=1e-6),torch.zeros_like(ws))[:,0]  # n,ny,nx
    flat=score.flatten().argsort(descending=True); picks=[]
    for fi in flat.tolist():
        f,yy,xx=np.unravel_index(fi,score.shape); y,x=int(yy)*STRIDE,int(xx)*STRIDE
        if any(abs(y-py)<CROP*0.75 and abs(x-px)<CROP*0.75 for _,py,px in picks): continue   # spatial NMS across all frames
        picks.append((int(f),y,x))
        if len(picks)==NHOT: break
    rng=np.random.RandomState(SEED+int(clip)); ctrl=[]
    while len(ctrl)<NCTRL:
        f=int(rng.randint(n)); y=int(rng.randint(0,TH-CROP+1)); x=int(rng.randint(0,TW-CROP+1))
        if md[f,0,y:y+CROP,x:x+CROP].sum()==0 and not any(f==pf and abs(y-py)<CROP and abs(x-px)<CROP for pf,py,px in picks+ctrl): ctrl.append((f,y,x))
    # ---- crops + stats
    rows=[]; heads=["GT right","warped input","origin output","composite (W out/O in)","|warped-GT| x4","mask: red=hard, yel=dil8 ring, cyan=soft"]
    recs=[]
    for tag,(f,y,x) in [(f"H{i+1}",p) for i,p in enumerate(picks)]+[(f"C{i+1}",p) for i,p in enumerate(ctrl)]:
        sl=(slice(None),slice(y,y+CROP),slice(x,x+CROP)); g=G[f][sl]; wv=warped[f][sl]; o=R[f][sl]; c=comp[f][sl]; om=out[f][sl]; hm=mb[f][sl]; dm=md[f][sl]; sm=soft[f][sl]
        e=eW[f][sl]; eo=eO[f][sl]; ec=eC[f][sl]
        r=dict(tag=tag,frame_idx=idx[f],sample=f,y=y,x=x,mask_hard_pct=round(100*hm.mean().item(),2),mask_dil_pct=round(100*dm.mean().item(),2),soft_pct=round(100*sm.mean().item(),2),
               dist_center_to_mask=round(float(dist[f,0,y+CROP//2,x+CROP//2]),1) if dist[f,0,y+CROP//2,x+CROP//2]<1e4 else None,
               err_out_warped=round(stat(e,om),4),err_out_origin=round(stat(eo,om),4),err_out_comp=round(stat(ec,om),4),
               psnr_warped=round(psnr(wv,g),2),psnr_origin=round(psnr(o,g),2),psnr_comp=round(psnr(c,g),2),
               frac_out_gt025_warped=round(100*stat((e>0.25).float(),om),2),frac_out_gt025_origin=round(100*stat((eo>0.25).float(),om),2))
        recs.append(r); P(f"[{clip}] {tag} f={idx[f]} y={y} x={x} maskHard%={r['mask_hard_pct']} dil%={r['mask_dil_pct']} soft%={r['soft_pct']} dist={r['dist_center_to_mask']} | out-mask |err| W={r['err_out_warped']} O={r['err_out_origin']} C={r['err_out_comp']} | PSNR W={r['psnr_warped']} O={r['psnr_origin']} C={r['psnr_comp']} | %px>0.25 W={r['frac_out_gt025_warped']} O={r['frac_out_gt025_origin']}")
        heat=(e*4).clamp(0,1).expand(3,-1,-1); ov=wv.clone(); ring=(dm-hm).clamp(0,1)
        ov=ov*(1-0.6*hm)+0.6*hm*torch.tensor([1.,0,0]).view(3,1,1); ov=ov*(1-0.5*ring)+0.5*ring*torch.tensor([1.,1,0]).view(3,1,1); ov=ov*(1-0.7*sm)+0.7*sm*torch.tensor([0,1.,1]).view(3,1,1)
        ims=[Image.fromarray(u8(t_)) for t_ in (g,wv,o,c,heat,ov)]
        lab=f"{tag} f{idx[f]} y{y} x{x} | |e|out W{r['err_out_warped']:.3f} O{r['err_out_origin']:.3f} C{r['err_out_comp']:.3f} | mask{r['mask_hard_pct']:.1f}%"
        rows.append((lab,ims))
    # ---- panel
    LM=0; hh=22; pad=4; Wp=len(heads)*(CROP+pad)+LM; Hp=hh+len(rows)*(CROP+hh+pad)
    panel=Image.new("RGB",(Wp,Hp),(30,30,30)); d=ImageDraw.Draw(panel)
    for j,hd in enumerate(heads): d.text((LM+j*(CROP+pad)+3,4),hd,font=FONTB,fill=(255,255,255))
    for i,(lab,ims) in enumerate(rows):
        yy=hh+i*(CROP+hh+pad); d.text((LM+3,yy+3),f"{clip} {lab}",font=FONTB,fill=(255,255,0) if lab.startswith("H") else (0,255,255))
        for j,im in enumerate(ims): panel.paste(im,(LM+j*(CROP+pad),yy+hh))
    pth=f"{OUT}/{clip}_hotspots.png"; assert not os.path.exists(pth), pth; panel.save(pth); P(f"[{clip}] wrote {pth}")
    # zoom: 2x nearest of the 4 image columns for the 3 hotspots
    z=Image.new("RGB",(4*(2*CROP+pad),NHOT*(2*CROP+hh)),(30,30,30)); dz=ImageDraw.Draw(z)
    for i,(lab,ims) in enumerate(rows[:NHOT]):
        yy=i*(2*CROP+hh); dz.text((3,yy+3),f"{clip} {lab}  [2x]  GT | warped | origin | composite",font=FONTB,fill=(255,255,0))
        for j,im in enumerate(ims[:4]): z.paste(im.resize((2*CROP,2*CROP),Image.NEAREST),(j*(2*CROP+pad),yy+hh))
    pz=f"{OUT}/{clip}_hotspots_zoom2x.png"; assert not os.path.exists(pz), pz; z.save(pz); P(f"[{clip}] wrote {pz}")
    json.dump(dict(clip=clip,n=n,step=STEP,align=(t0,l0,alps),crop=(top,left),crops=recs,log=log),open(f"{RUN}/{clip}_hotspots.json","w"),indent=1)
    P(f"[{clip}] done in {time.time()-t:.0f}s")

for c in sys.argv[1:]: run(c)
