"""SLOT BUDGET - PASS B: same UNet benchmark as scripts/distill/bench2.py (unmodified copy of its
setup, warmup and timing protocol) PLUS the per-slot CUDA-event module timer that
inpainting_inference.py exposes as --module_profile_json / --module_profile_include
(utils/module_timing.install_cuda_module_timer, default include '*.attn1').

Here the include list is restricted to the FIVE level-0 spatial attn1 slots that the shipped
deliverable replaces, so the hooks perturb only what we are attributing:
  down_blocks.0.attentions.{0,1}.transformer_blocks.0.attn1
  up_blocks.3.attentions.{0,1,2}.transformer_blocks.0.attn1

Two phases per process:
  clean   : 3 untimed warmup forwards, then 10 timed forwards (no hooks)  -> the published number
  hooked  : hooks installed, 1 warmup, events cleared, then 10 timed forwards -> per-slot times
Module time is INCLUSIVE of the slot's children (for a replaced slot that is the whole
GatedResidualMambaSelfAttention adapter; the re-parented attn1.origin_attn is not hooked and, at
mamba_gate=1.0 with reference_disabled, is never evaluated - per_fwd_calls proves it).

usage: bench_slots_v1.py LABEL INCLUDE EXCLUDE   (env: MODE, GATE, H, W, BS, DS, EXP, BIDIR, CKPT, SB_JSON)
"""
import os, sys, time, json, torch

REPO = "/home/kawa/master_project/StereoCrafter"
sys.path.insert(0, REPO)

LABEL, INC, EXC = sys.argv[1], sys.argv[2], sys.argv[3]
MODE = os.environ.get("MODE", "gated_residual")
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
from utils.module_timing import install_cuda_module_timer, write_cuda_module_timing

pre = f"{REPO}/weights/stable-video-diffusion-img2vid-xt-1-1/"
up = f"{REPO}/weights/StereoCrafter/"
dt = torch.float16
ie = CLIPVisionModelWithProjection.from_pretrained(pre, subfolder="image_encoder", variant="fp16", torch_dtype=dt)
vae = AutoencoderKLTemporalDecoder.from_pretrained(pre, subfolder="vae", variant="fp16", torch_dtype=dt)
unet = UNetSpatioTemporalConditionModel.from_pretrained(up, subfolder="unet_diffusers", low_cpu_mem_usage=True, torch_dtype=dt)
pipe = P.from_pretrained(pre, image_encoder=ie, vae=vae, unet=unet, torch_dtype=dt)
ck = os.environ.get("CKPT")
if ck:
    sd = torch.load(ck, map_location="cpu")["model"]
    materialize_mamba_time_embed_proj_from_state_dict(pipe.unet, sd)
    pipe.unet.to(dtype=torch.float32); pipe.unet.load_state_dict(sd, strict=False)
pipe.unet.to(dtype=dt); u = pipe.unet.to("cuda").eval()
if ck and os.environ.get("GATE"):
    from blocks.mamba_diffusers_adapter import set_gated_mamba_gate
    set_gated_mamba_gate(u, float(os.environ["GATE"]))

# ---- instrumentation identical to bench2.py: count what really runs ----
counts = {"origin_attn": 0, "mamba_core": 0, "plain_attn1": 0}
gates = []
for n, m in u.named_modules():
    cn = m.__class__.__name__
    if n.endswith(".origin_attn"):
        m.register_forward_hook(lambda *a: counts.__setitem__("origin_attn", counts["origin_attn"] + 1))
    if n.endswith(".fwd.core") or n.endswith(".bwd.core"):
        m.register_forward_hook(lambda *a: counts.__setitem__("mamba_core", counts["mamba_core"] + 1))
    if n.endswith(".attn1") and "temporal" not in n and cn == "Attention":
        m.register_forward_hook(lambda *a: counts.__setitem__("plain_attn1", counts["plain_attn1"] + 1))
    if hasattr(m, "mamba_gate"): gates.append(float(m.mamba_gate))

SLOTS = [f"down_blocks.0.attentions.{i}.transformer_blocks.0.attn1" for i in (0, 1)] + \
        [f"up_blocks.3.attentions.{i}.transformer_blocks.0.attn1" for i in (0, 1, 2)]
slot_classes = {n: m.__class__.__name__ for n, m in u.named_modules() if n in SLOTS}

F = 14
H = int(os.environ.get("H", 576)); W = int(os.environ.get("W", 1024))
BS = int(os.environ.get("BS", 1))
inp = torch.cat([torch.randn(BS, F, 8, H // 8, W // 8, device="cuda", dtype=dt),
                 torch.randn(BS, F, 1, H // 8, W // 8, device="cuda", dtype=dt)], 2)
t = torch.tensor([1.0], device="cuda"); enc = torch.randn(BS, 1, 1024, device="cuda", dtype=dt)
add = torch.tensor([[7.0, 127.0, 0.0]] * BS, device="cuda", dtype=dt)

with torch.no_grad():
    # ---- phase 1: clean (published protocol) ----
    for _ in range(3): u(inp, t, encoder_hidden_states=enc, added_time_ids=add)
    for k in counts: counts[k] = 0
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t0 = time.time()
    for _ in range(10): u(inp, t, encoder_hidden_states=enc, added_time_ids=add)
    torch.cuda.synchronize(); sec_clean = (time.time() - t0) / 10
    peak_clean = round(torch.cuda.max_memory_allocated() / 2 ** 20)
    peak_res_clean = round(torch.cuda.max_memory_reserved() / 2 ** 20)
    for k in counts: counts[k] //= 10  # per forward

    # ---- phase 2: hooked (per-slot attribution) ----
    state, handles = install_cuda_module_timer(u, include=",".join(SLOTS), default_include=["*.attn1"])
    assert len(state["modules"]) == 5, f"expected 5 hooked slots, got {sorted(state['modules'])}"
    u(inp, t, encoder_hidden_states=enc, added_time_ids=add)      # warmup with hooks live
    torch.cuda.synchronize()
    for rec in state["modules"].values(): rec["events"].clear()   # drop the warmup events
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(10): u(inp, t, encoder_hidden_states=enc, added_time_ids=add)
    torch.cuda.synchronize(); sec_hooked = (time.time() - t0) / 10

JSON = os.environ.get("SB_JSON") or f"{REPO}/scripts/distill/runs/slotbudget/profiles/{LABEL}.json"
payload = write_cuda_module_timing(state, JSON, metadata={
    "label": LABEL, "include": INC, "exclude": EXC, "mode": MODE, "H": H, "W": W, "batch": BS,
    "frames": F, "dstate": os.environ.get("DS"), "expand": os.environ.get("EXP"),
    "bidir": os.environ.get("BIDIR"), "ckpt": ck, "gate": os.environ.get("GATE"),
    "timedForwards": 10, "warmupForwards": 3, "secCleanNoHooks": sec_clean, "secHooked": sec_hooked,
    "peakMiBAllocated": peak_clean, "perFwdCalls": counts, "slotClasses": slot_classes,
    "moduleTimeSemantics": "inclusive forward time of the module named (CUDA events, pre/post forward hook)",
})
for h in handles: h.remove()
slots = {m["name"]: {"cls": m["class"], "calls": m["calls"], "avgMs": round(m["avgMs"], 4)}
         for m in payload["modules"]}
print("RESULT " + json.dumps({
    "label": LABEL, "mode": MODE, "H": H, "W": W, "tokens": (H // 8) * (W // 8), "batch": BS,
    "sec_clean": round(sec_clean, 4), "sec_hooked": round(sec_hooked, 4),
    "peak_MiB": peak_clean, "peak_reserved_MiB": peak_res_clean,
    "per_fwd_calls": counts, "gates": sorted(set(round(g, 3) for g in gates)),
    "slot_classes": slot_classes, "slots": slots,
    "slot_sum_avgMs": round(sum(v["avgMs"] for v in slots.values()), 4), "json": JSON}))
