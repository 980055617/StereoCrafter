"""CPU-only: how far did the 25 trainable attn1 tensors move from origin after 1 and 2 epochs?"""
import sys, torch
from safetensors.torch import load_file
PREF = ["down_blocks.0.attentions.0.transformer_blocks.0.attn1.",
        "down_blocks.0.attentions.1.transformer_blocks.0.attn1.",
        "up_blocks.3.attentions.0.transformer_blocks.0.attn1.",
        "up_blocks.3.attentions.1.transformer_blocks.0.attn1.",
        "up_blocks.3.attentions.2.transformer_blocks.0.attn1."]
origin = load_file("weights/StereoCrafter/unet_diffusers/diffusion_pytorch_model.safetensors")
okeys = [k for k in origin if any(k.startswith(p) for p in PREF)]
print("origin matching keys:", len(okeys))

def find_sd(obj, depth=0):
    if isinstance(obj, dict):
        if any(isinstance(v, torch.Tensor) for v in obj.values()) and any(k.startswith("down_blocks") or "down_blocks" in k for k in obj):
            return obj
        for k, v in obj.items():
            r = find_sd(v, depth+1)
            if r is not None:
                print("  state dict found under key:", k)
                return r
    return None

for path in sys.argv[1:]:
    ck = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    if isinstance(ck, dict):
        print(path, "top keys:", [ (k, type(v).__name__) for k, v in ck.items() if not isinstance(v, torch.Tensor)][:20])
    sd = find_sd(ck)
    if sd is None:
        print("no state dict"); continue
    keys = list(sd.keys())
    print("  n tensors:", len(keys), "sample:", keys[:3])
    cnt = 0; tot_rel = []
    for k in okeys:
        cands = [kk for kk in keys if kk.endswith(k) or kk == k]
        if not cands:
            continue
        w = sd[cands[0]].float(); o = origin[k].float()
        if w.shape != o.shape:
            print("  shape mismatch", k, w.shape, o.shape); continue
        d = (w - o)
        rel = d.norm() / o.norm()
        tot_rel.append(rel.item()); cnt += 1
        print(f"  {k:70s} dtype={sd[cands[0]].dtype} |w|={o.norm():.3f} |dw|={d.norm():.4f} rel={rel:.4f} max|dw|={d.abs().max():.2e} mean|w|={o.abs().mean():.2e} mean|dw|={d.abs().mean():.2e}")
    if cnt:
        print(f"  matched {cnt}/{len(okeys)}; mean rel delta {sum(tot_rel)/len(tot_rel):.4f}")
    # also: did any NON-trainable tensor change?
    changed = 0; checked = 0
    for k in list(origin.keys())[::37]:
        cands = [kk for kk in keys if kk.endswith(k) or kk == k]
        if not cands: continue
        checked += 1
        if not torch.equal(sd[cands[0]].float(), origin[k].float()):
            if not any(k.startswith(p) for p in PREF):
                changed += 1
    print(f"  frozen-tensor spot check: {changed} of {checked} sampled non-attn1 tensors differ from origin")
