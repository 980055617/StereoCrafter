"""Write a train_state .pt whose Mamba attn1 modules are replaced by standalone-distilled weights.
usage: python inject_distilled.py <base_train_state.pt> <distilled.pt from distill_standalone SAVE> <out.pt>
Keeps everything else (incl. mamba_gate, origin_attn) from the base checkpoint.
"""
import sys, torch
base, dist, out = sys.argv[1:4]
b = torch.load(base, map_location="cpu"); d = torch.load(dist, map_location="cpu")["model"]
m = b["model"]; n_rep = 0; n_new = 0
for k, v in d.items():
    if k in m:
        m[k] = v.to(m[k].dtype); n_rep += 1
    else:
        m[k] = v; n_new += 1
print(f"replaced={n_rep} added={n_new} of {len(d)} distilled tensors; modules={sorted({k.split('.attn1.')[0] for k in d})}")
torch.save({"model": m, "epoch": b.get("epoch"), "distilled_from": dist}, out); print("wrote", out)
