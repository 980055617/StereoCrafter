"""CPU-only: how far did the 25 trainable attn1 tensors move in the control run?
Compares origin fp16 safetensors vs epoch-1 / epoch-2 bf16 checkpoints.
"""
import json, math, sys, torch
from safetensors.torch import load_file
ORIGIN = "weights/StereoCrafter/unet_diffusers/diffusion_pytorch_model.safetensors"
RUN = "weights/GTfinetune_v2_originattn_control/MambaCrafter_20260925_143200"
KEEP = ["down_blocks.0.attentions.0.transformer_blocks.0.attn1.",
        "down_blocks.0.attentions.1.transformer_blocks.0.attn1.",
        "up_blocks.3.attentions.0.transformer_blocks.0.attn1.",
        "up_blocks.3.attentions.1.transformer_blocks.0.attn1.",
        "up_blocks.3.attentions.2.transformer_blocks.0.attn1."]
orig = load_file(ORIGIN)
e1 = torch.load(f"{RUN}/train_state_epoch000001.pt", map_location="cpu", weights_only=False)["model"]
e2 = torch.load(f"{RUN}/train_state_epoch000002.pt", map_location="cpu", weights_only=False)["model"]
assert set(orig) == set(e1) == set(e2), (len(orig), len(e1), len(e2))
train_keys = [k for k in orig if any(k.startswith(p) for p in KEEP)]
print("n trainable keys", len(train_keys))
# 1) frozen base: is ckpt == bf16(origin) bit-exactly? and how big is the fp16->bf16 round trip?
frozen_mismatch = 0; frozen_n = 0; rt_num = 0.0; rt_den = 0.0; worst_rt = (0.0, None)
for k, v in orig.items():
    if k in train_keys: continue
    frozen_n += 1
    vb = v.to(torch.bfloat16)
    if not torch.equal(vb, e1[k]) or not torch.equal(vb, e2[k]):
        frozen_mismatch += 1
    d = (vb.float() - v.float()); num = d.pow(2).sum().item(); den = v.float().pow(2).sum().item()
    rt_num += num; rt_den += den
    rel = math.sqrt(num / max(den, 1e-30))
    if rel > worst_rt[0]: worst_rt = (rel, k)
print(f"frozen tensors: {frozen_n}, mismatching bf16(origin) in e1/e2: {frozen_mismatch}")
print(f"fp16->bf16 round-trip of frozen base: global relF = {math.sqrt(rt_num/rt_den):.3e}, worst tensor relF = {worst_rt[0]:.3e} ({worst_rt[1]})")
# 2) trainable tensors: movement
def ulp(x):  # bf16 ulp for each element (8 bits of precision incl. implicit)
    e = torch.floor(torch.log2(x.abs().clamp_min(1e-30)))
    return torch.pow(2.0, e - 7)
rows = []
tot = {"d1": 0.0, "d2": 0.0, "d12": 0.0, "den": 0.0}
for k in train_keys:
    w0f = orig[k].float(); w0 = orig[k].to(torch.bfloat16).float(); w1 = e1[k].float(); w2 = e2[k].float()
    d1 = w1 - w0; d2 = w2 - w0; d12 = w2 - w1
    den = w0.pow(2).sum().item()
    rel1 = math.sqrt(d1.pow(2).sum().item()/den); rel2 = math.sqrt(d2.pow(2).sum().item()/den); rel12 = math.sqrt(d12.pow(2).sum().item()/den)
    relrt = math.sqrt((w0 - w0f).pow(2).sum().item()/den)
    changed1 = (d1 != 0).float().mean().item(); changed12 = (d12 != 0).float().mean().item()
    u = ulp(w0)
    ulps1 = (d1.abs() / u); ulps12 = (d12.abs() / u)
    cos = torch.nn.functional.cosine_similarity(d1.flatten(), d12.flatten(), dim=0).item() if d12.abs().sum() > 0 else float("nan")
    # sign consistency of the 2 deltas across elements
    tot["d1"] += d1.pow(2).sum().item(); tot["d2"] += d2.pow(2).sum().item(); tot["d12"] += d12.pow(2).sum().item(); tot["den"] += den
    rows.append(dict(key=k, shape=list(orig[k].shape), w_rms=math.sqrt(den/w0.numel()), w_absmax=w0.abs().max().item(),
        relF_e1=rel1, relF_e2=rel2, relF_e1_to_e2=rel12, relF_fp16_to_bf16_roundtrip=relrt,
        frac_changed_e1=changed1, frac_changed_e1_to_e2=changed12,
        mean_ulps_e1=ulps1.mean().item(), max_ulps_e1=ulps1.max().item(), p99_ulps_e1=ulps1.flatten().kthvalue(int(0.99*ulps1.numel())).values.item(),
        max_abs_delta_e1=d1.abs().max().item(), mean_abs_delta_e1=d1.abs().mean().item(),
        cos_delta1_delta12=cos))
for r in rows:
    print(f"{r['key']:<70s} {str(r['shape']):>12s} wrms={r['w_rms']:.4f} relF_e1={r['relF_e1']:.3e} relF_e2={r['relF_e2']:.3e} relF_e1->e2={r['relF_e1_to_e2']:.3e} rt={r['relF_fp16_to_bf16_roundtrip']:.2e} changed_e1={r['frac_changed_e1']:.3f} changed_e1->e2={r['frac_changed_e1_to_e2']:.3f} ulps_e1(mean/p99/max)={r['mean_ulps_e1']:.2f}/{r['p99_ulps_e1']:.1f}/{r['max_ulps_e1']:.0f} max|d1|={r['max_abs_delta_e1']:.2e} cos(d1,d12)={r['cos_delta1_delta12']:.3f}")
print(f"GLOBAL over 25 tensors: relF_e1={math.sqrt(tot['d1']/tot['den']):.3e} relF_e2={math.sqrt(tot['d2']/tot['den']):.3e} relF_e1->e2={math.sqrt(tot['d12']/tot['den']):.3e}")
json.dump(rows, open("scripts/distill/runs/diag_trainer/weight_delta_control_run.json", "w"), indent=1)
