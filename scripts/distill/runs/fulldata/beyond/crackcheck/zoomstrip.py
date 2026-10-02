"""4x zoom of a region: left-eye input | warped | 'GT' (train TR) | raw mask x3 | origin | composite. usage: zoomstrip.py clip frame_idx y x size"""
import sys, json, numpy as np, torch, torch.nn.functional as F
from decord import VideoReader, cpu
from PIL import Image, ImageDraw, ImageFont
ROOT="/home/kawa/master_project/StereoCrafter"; OUT=f"{ROOT}/outputs/fulldata/beyond/crackcheck"; RUN=f"{ROOT}/scripts/distill/runs/fulldata/beyond/crackcheck"; TH,TW=576,1024
clip,fi,y,x,S=sys.argv[1],int(sys.argv[2]),int(sys.argv[3]),int(sys.argv[4]),int(sys.argv[5]); J=json.load(open(f"{RUN}/{clip}_hotspots.json")); t0,l0,_=J["align"]; top,left=J["crop"]
t=VideoReader(f"{ROOT}/video_data/train/{clip}_train.mp4",ctx=cpu(0))[fi].asnumpy(); H,W=t.shape[0]//2,t.shape[1]//2; G=t[t0:t0+TH,W+l0:W+l0+TW]
s=VideoReader(f"{ROOT}/video_data/splatting/{clip}_splatting_results.mp4",ctx=cpu(0))[fi].asnumpy(); h,w=s.shape[0]//2,s.shape[1]//2
inL=s[top:top+TH,left:left+TW]; M=s[h+top:h+top+TH,left:left+TW]; Wp=s[h+top:h+top+TH,w+left:w+left+TW]
v=VideoReader(f"{ROOT}/outputs/fulldata/clips/{clip}_origin/{clip}_inpainting_results_sbs.mp4",ctx=cpu(0))[fi].asnumpy(); R=v[:,v.shape[1]//2:]
mb=torch.from_numpy((M.mean(2)>127.5).astype(np.float32))[None,None]; md=F.max_pool2d(mb,17,1,8); m=F.avg_pool2d(md,17,1,8)[0,0].numpy()[...,None]
C=(Wp*(1-m)+R*m).astype(np.uint8); Mx=np.clip(M.astype(np.int32)*3,0,255).astype(np.uint8)
FONT=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",14); cols=[("left-eye input",inL),("warped",Wp),("train TR ('GT')",G),("mask x3",Mx),("origin",R),("composite",C)]
im=Image.new("RGB",(len(cols)*(4*S+4),4*S+22),(30,30,30)); d=ImageDraw.Draw(im); d.text((3,3),f"{clip} f={fi} y={y} x={x} {S}px @4x: "+" | ".join(c for c,_ in cols),font=FONT,fill=(255,255,0))
for j,(_,a) in enumerate(cols): im.paste(Image.fromarray(np.ascontiguousarray(a[y:y+S,x:x+S])).resize((4*S,4*S),Image.NEAREST),(j*(4*S+4),22))
p=f"{OUT}/{clip}_zoom_f{fi}_y{y}_x{x}.png"; im.save(p); print("wrote",p)
