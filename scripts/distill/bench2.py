"""Instrumented UNet benchmark. Counts what actually executed so a gate=0
double-execution (Mamba AND reference attention both running) can never
hide again. Usage: bench2.py LABEL INCLUDE EXCLUDE  (env: MODE, GATE, H, W, DS, EXP, BIDIR, CKPT)"""
import os, sys, time, json, torch
LABEL, INC, EXC = sys.argv[1], sys.argv[2], sys.argv[3]
MODE = os.environ.get("MODE", "gated_residual")      # gated_residual | mamba
os.environ["MAMBA_SELF_ATTN_REPLACEMENT"] = MODE
os.environ["MAMBA_SELF_ATTN_INCLUDE"] = INC if INC else "__nomatch__"
if EXC: os.environ["MAMBA_SELF_ATTN_EXCLUDE"] = EXC
if os.environ.get("GATE"): os.environ["MAMBA_SELF_ATTN_INITIAL_GATE"] = os.environ["GATE"]
if os.environ.get("DS"):   os.environ["MAMBA_SELF_ATTN_D_STATE"] = os.environ["DS"]
if os.environ.get("EXP"):  os.environ["MAMBA_SELF_ATTN_EXPAND"] = os.environ["EXP"]
if os.environ.get("BIDIR"):os.environ["MAMBA_BIDIRECTIONAL_MODE"] = os.environ["BIDIR"]
os.environ["MAMBA_ADAPTER_LOG"] = "1"

from transformers import CLIPVisionModelWithProjection
from diffusers import AutoencoderKLTemporalDecoder, UNetSpatioTemporalConditionModel
from pipelines.mamba_stereo_video_inpainting_pipeline import MambaStableVideoDiffusionInpaintingPipeline as P
from blocks.mamba_diffusers_adapter import materialize_mamba_time_embed_proj_from_state_dict
pre="weights/stable-video-diffusion-img2vid-xt-1-1/"; up="weights/StereoCrafter/"; dt=torch.float16
ie=CLIPVisionModelWithProjection.from_pretrained(pre,subfolder="image_encoder",variant="fp16",torch_dtype=dt)
vae=AutoencoderKLTemporalDecoder.from_pretrained(pre,subfolder="vae",variant="fp16",torch_dtype=dt)
unet=UNetSpatioTemporalConditionModel.from_pretrained(up,subfolder="unet_diffusers",low_cpu_mem_usage=True,torch_dtype=dt)
pipe=P.from_pretrained(pre,image_encoder=ie,vae=vae,unet=unet,torch_dtype=dt)
ck=os.environ.get("CKPT")
if ck:
    sd=torch.load(ck,map_location="cpu")["model"]; materialize_mamba_time_embed_proj_from_state_dict(pipe.unet,sd)
    pipe.unet.to(dtype=torch.float32); pipe.unet.load_state_dict(sd,strict=False)
pipe.unet.to(dtype=dt); u=pipe.unet.to("cuda").eval()
if ck and os.environ.get("GATE"):
    from blocks.mamba_diffusers_adapter import set_gated_mamba_gate
    set_gated_mamba_gate(u, float(os.environ["GATE"]))   # checkpoint buffer would otherwise win

# ---- instrumentation: count what really runs ----
counts={"origin_attn":0,"mamba_core":0,"plain_attn1":0}
gates=[]
for n,m in u.named_modules():
    cn=m.__class__.__name__
    if n.endswith(".origin_attn"):
        m.register_forward_hook(lambda *a: counts.__setitem__("origin_attn",counts["origin_attn"]+1))
    if n.endswith(".fwd.core") or n.endswith(".bwd.core"):
        m.register_forward_hook(lambda *a: counts.__setitem__("mamba_core",counts["mamba_core"]+1))
    if n.endswith(".attn1") and "temporal" not in n and cn=="Attention":
        m.register_forward_hook(lambda *a: counts.__setitem__("plain_attn1",counts["plain_attn1"]+1))
    if hasattr(m,"mamba_gate"): gates.append(float(m.mamba_gate))

F=14; H=int(os.environ.get("H",576))//8; W=int(os.environ.get("W",1024))//8
BS=int(os.environ.get("BS",1))
inp=torch.cat([torch.randn(BS,F,8,H,W,device="cuda",dtype=dt),torch.randn(BS,F,1,H,W,device="cuda",dtype=dt)],2)
t=torch.tensor([1.0],device="cuda"); enc=torch.randn(BS,1,1024,device="cuda",dtype=dt)
add=torch.tensor([[7.0,127.0,0.0]]*BS,device="cuda",dtype=dt)
with torch.no_grad():
    for _ in range(3): u(inp,t,encoder_hidden_states=enc,added_time_ids=add)
    for k in counts: counts[k]=0
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t0=time.time()
    for _ in range(10): u(inp,t,encoder_hidden_states=enc,added_time_ids=add)
    torch.cuda.synchronize(); s=(time.time()-t0)/10
for k in counts: counts[k]//=10   # per forward
print("RESULT "+json.dumps({"label":LABEL,"mode":MODE,"tokens":H*W,"batch":BS,"sec":round(s,4),
      "peak_MiB":round(torch.cuda.max_memory_allocated()/2**20),
      "per_fwd_calls":counts,"gates":sorted(set(round(g,3) for g in gates))}))
