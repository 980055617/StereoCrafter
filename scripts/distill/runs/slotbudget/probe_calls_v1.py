"""Probe: how many times is each attn1 module actually called per UNet forward, and is the
same module object reachable under several names?  Explains bench2.py's per_fwd_calls numbers."""
import os, sys, json, collections, torch
REPO="/home/kawa/master_project/StereoCrafter"; sys.path.insert(0,REPO)
INC=sys.argv[1]
os.environ["MAMBA_SELF_ATTN_REPLACEMENT"]="gated_residual"
os.environ["MAMBA_SELF_ATTN_INCLUDE"]=INC
os.environ["MAMBA_SELF_ATTN_EXCLUDE"]="__nomatch__"
os.environ["MAMBA_SELF_ATTN_D_STATE"]="128"; os.environ["MAMBA_SELF_ATTN_EXPAND"]="1"
os.environ["MAMBA_BIDIRECTIONAL_MODE"]="fwd"; os.environ["MAMBA_SELF_ATTN_INITIAL_GATE"]="1.0"
from transformers import CLIPVisionModelWithProjection
from diffusers import AutoencoderKLTemporalDecoder, UNetSpatioTemporalConditionModel
from pipelines.mamba_stereo_video_inpainting_pipeline import MambaStableVideoDiffusionInpaintingPipeline as P
pre=f"{REPO}/weights/stable-video-diffusion-img2vid-xt-1-1/"; up=f"{REPO}/weights/StereoCrafter/"; dt=torch.float16
ie=CLIPVisionModelWithProjection.from_pretrained(pre,subfolder="image_encoder",variant="fp16",torch_dtype=dt)
vae=AutoencoderKLTemporalDecoder.from_pretrained(pre,subfolder="vae",variant="fp16",torch_dtype=dt)
unet=UNetSpatioTemporalConditionModel.from_pretrained(up,subfolder="unet_diffusers",low_cpu_mem_usage=True,torch_dtype=dt)
pipe=P.from_pretrained(pre,image_encoder=ie,vae=vae,unet=unet,torch_dtype=dt)
u=pipe.unet.to(dtype=dt).to("cuda").eval()
names=collections.defaultdict(int); ids=collections.defaultdict(list)
for n,m in u.named_modules():
    if n.endswith(".attn1") and "temporal" not in n:
        ids[id(m)].append(n)
        m.register_forward_hook(lambda mod,i,o,_n=n: names.__setitem__(_n,names[_n]+1))
print("distinct spatial attn1 objects:",len(ids),"names:",sum(len(v) for v in ids.values()))
for k,v in ids.items():
    if len(v)>1: print("  SHARED OBJECT under",v)
F=14;H=576//8;W=1024//8;BS=2
inp=torch.cat([torch.randn(BS,F,8,H,W,device="cuda",dtype=dt),torch.randn(BS,F,1,H,W,device="cuda",dtype=dt)],2)
t=torch.tensor([1.0],device="cuda");enc=torch.randn(BS,1,1024,device="cuda",dtype=dt)
add=torch.tensor([[7.0,127.0,0.0]]*BS,device="cuda",dtype=dt)
with torch.no_grad(): u(inp,t,encoder_hidden_states=enc,added_time_ids=add)
print("PROBE "+json.dumps({"include":INC,"per_name_calls":dict(sorted(names.items())),"total":sum(names.values())}))
