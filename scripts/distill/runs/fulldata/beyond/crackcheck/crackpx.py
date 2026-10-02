"""Crack-pixel census: pixels in the warped right eye that are near-black while their horizontal 7-neighbourhood is bright and GT is not
dark there (=splat holes/cracks). Where do they sit relative to the 0.5-threshold mask, its 8px dilation, and the feathered composite weight?
usage: python crackpx.py clip ...   (writes {clip}_crackpx.json + a marked crop PNG)"""
import sys, os, json, numpy as np, torch, torch.nn.functional as F
from decord import VideoReader, cpu
from PIL import Image, ImageDraw, ImageFont
ROOT="/home/kawa/master_project/StereoCrafter"; OUT=f"{ROOT}/outputs/fulldata/beyond/crackcheck"; RUN=f"{ROOT}/scripts/distill/runs/fulldata/beyond/crackcheck"; TH,TW=576,1024; STEP=4
FONT=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",14)
def load_u8(p): vr=VideoReader(p,ctx=cpu(0)); return vr.get_batch(list(range(0,len(vr),STEP))).asnumpy()
def to_t(a): return torch.from_numpy(np.ascontiguousarray(a)).permute(0,3,1,2).float()/255.
def lum(x): return (0.299*x[:,0:1]+0.587*x[:,1:2]+0.114*x[:,2:3])
for clip in sys.argv[1:]:
    J=json.load(open(f"{RUN}/{clip}_hotspots.json")); t0,l0,_=J["align"]; top,left=J["crop"]
    tile=load_u8(f"{ROOT}/video_data/train/{clip}_train.mp4"); H,W=tile.shape[1]//2,tile.shape[2]//2; G=to_t(tile[:,t0:t0+TH,W+l0:W+l0+TW]); del tile
    sp=load_u8(f"{ROOT}/video_data/splatting/{clip}_splatting_results.mp4"); h,w=sp.shape[1]//2,sp.shape[2]//2
    mask=to_t(sp[:,h+top:h+top+TH,left:left+TW]).mean(1,keepdim=True); Wp=to_t(sp[:,h+top:h+top+TH,w+left:w+left+TW]); del sp
    v=to_t(load_u8(f"{ROOT}/outputs/fulldata/clips/{clip}_origin/{clip}_inpainting_results_sbs.mp4")); R=v[:,:,:,v.shape[3]//2:]
    n=min(len(G),len(Wp),len(R)); G,Wp,mask,R=G[:n],Wp[:n],mask[:n],R[:n]
    yw,yg,yo=lum(Wp),lum(G),lum(R)
    hmax=F.max_pool2d(yw,(1,7),1,(0,3)); gmin=-F.max_pool2d(-yg,7,1,3)
    crack=((yw<0.08)&(hmax>0.35)&(gmin>0.15)).float()
    mb=(mask>0.5).float(); md=F.max_pool2d(mb,17,1,8); m=F.avg_pool2d(md,17,1,8)
    tot=crack.sum().item(); res=dict(clip=clip,n=n,crack_px_total=int(tot),crack_px_per_frame=tot/n,crack_ppm_of_frame=1e6*tot/(n*TH*TW))
    def frac(w): return round(100*(crack*w).sum().item()/max(tot,1),2)
    res.update(inside_hard_mask_pct=frac(mb),inside_dil8_pct=frac(md),outside_dil8_pct=frac(1-md),composite_weight_lt05_pct=frac((m<0.5).float()),composite_weight_lt01_pct=frac((m<0.1).float()))
    # mask value at crack px outside dil8
    o=crack*(1-md); res["outside_dil8_maskval_bins_pct"]={f"[{a},{b})":round(100*(o*((mask>=a)&(mask<b)).float()).sum().item()/max(o.sum().item(),1),1) for a,b in [(0,0.005),(0.005,0.02),(0.02,0.1),(0.1,0.25),(0.25,0.5),(0.5,2)]}
    # would alternative masks cover them?
    alt={}
    for thr in (0.5,0.25,0.1,0.02):
        for d in (0,8,16):
            mm=(mask>thr).float(); mm=F.max_pool2d(mm,2*d+1,1,d) if d else mm
            alt[f"thr{thr}_dil{d}"]=dict(covered_pct=frac(mm),mask_area_pct=round(100*mm.mean().item(),2))
    res["alt_masks"]=alt
    # does origin fix them? origin lum at crack px > 0.2
    res["origin_bright_at_crack_pct"]=frac((yo>0.2).float()); res["warped_err_at_crack"]=round(((Wp-G).abs().mean(1,keepdim=True)*crack).sum().item()/max(tot,1),3); res["origin_err_at_crack"]=round(((R-G).abs().mean(1,keepdim=True)*crack).sum().item()/max(tot,1),3)
    # per-frame count of shipped (composite-weight<0.5) crack px, and the worst frame
    ship=(crack*(m<0.5).float()).sum((1,2,3)); fi=int(ship.argmax()); res["shipped_crack_px_per_frame"]=[int(x) for x in ship.tolist()]; res["worst_frame_sample"]=fi
    # connected-component size of shipped crack px in worst frame (are they lines or dots?)
    from scipy import ndimage
    lab,nc=ndimage.label((crack*(m<0.5))[fi,0].numpy()>0); sizes=np.bincount(lab.ravel())[1:] if nc else np.array([])
    res["worst_frame_components"]=dict(n=int(nc),max_size=int(sizes.max()) if nc else 0,ge5px=int((sizes>=5).sum()) if nc else 0,ge20px=int((sizes>=20).sum()) if nc else 0)
    print(json.dumps(res),flush=True); json.dump(res,open(f"{RUN}/{clip}_crackpx.json","w"),indent=1)
    # marked crop: 256x256 around the densest shipped-crack window in the worst frame: warped | warped with crack px magenta, mask overlay | composite | GT
    if nc:
        dens=F.avg_pool2d((crack*(m<0.5))[fi:fi+1],256,16)[0,0]; yy,xx=np.unravel_index(int(dens.argmax()),dens.shape); y,x=int(yy)*16,int(xx)*16
        sl=(slice(None),slice(y,y+256),slice(x,x+256)); wv=Wp[fi][sl]; c=(Wp*(1-m)+R*m)[fi][sl]; g=G[fi][sl]; ck=crack[fi][sl]; hm=mb[fi][sl]; dm=md[fi][sl]
        ov=wv*(1-0.5*hm)+0.5*hm*torch.tensor([1.,0,0]).view(3,1,1); ov=ov*(1-0.4*(dm-hm))+0.4*(dm-hm)*torch.tensor([1.,1,0]).view(3,1,1); ov=ov*(1-ck)+ck*torch.tensor([1.,0,1.]).view(3,1,1)
        u8=lambda t:(t.clamp(0,1)*255).round().byte().permute(1,2,0).numpy()
        im=Image.new("RGB",(4*(512+4),512+22),(30,30,30)); d=ImageDraw.Draw(im); d.text((3,3),f"{clip} worst shipped-crack frame f={fi*STEP} y={y} x={x} [2x]: warped | crack px=magenta, red=hard mask, yel=dil8 ring | composite | GT   shipped crack px in crop={int(ck.sum())}",font=FONT,fill=(255,0,255))
        for j,t_ in enumerate((wv,ov,c,g)): im.paste(Image.fromarray(u8(t_)).resize((512,512),Image.NEAREST),(j*516,22))
        p=f"{OUT}/{clip}_crackpx_worst.png"; assert not os.path.exists(p); im.save(p); print("wrote",p,flush=True)
