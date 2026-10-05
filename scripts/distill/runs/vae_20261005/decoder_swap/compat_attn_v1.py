#!/usr/bin/env python
"""vae_20261005 / decoder_swap -- STEP 1 addendum (CPU): the 16 encoder mid-block attention tensors that compat_v1.py could not
pair by name (sd-vae-ft-mse / ft-ema use the legacy diffusers names query/key/value/proj_attn; SVD uses to_q/to_k/to_v/to_out.0).
Same comparison rule as compat_v1.py, after the legacy->current name mapping that diffusers itself applies on load
(the 1x1 legacy linear weights have the same [out,in] shape as the current Linear weights).
usage: python compat_attn_v1.py <out_json>
"""
import json, os, sys
import torch
from safetensors import safe_open
OUTJ = sys.argv[1]
assert not os.path.exists(OUTJ)
SVD = "/mnt/ssd_data/stereocrafter_weights/stable-video-diffusion-img2vid-xt-1-1/vae/diffusion_pytorch_model.safetensors"
O = {"ftmse": "/home/kawa/master_project/third_party/DiffuEraser/weights/sd-vae-ft-mse/diffusion_pytorch_model.safetensors",
     "ftema": "/mnt/ssd_data/vae_20261005/decoder_swap/hf_home/hub/models--stabilityai--sd-vae-ft-ema/snapshots/f04b2c4b98319346dad8c65879f680b1997b204a/diffusion_pytorch_model.safetensors"}
MAP = {"query": "to_q", "key": "to_k", "value": "to_v", "proj_attn": "to_out.0"}
pre = "encoder.mid_block.attentions.0."
with safe_open(SVD, "pt") as f:
    s = {k: f.get_tensor(k) for k in f.keys() if k.startswith(pre)}
res = {}
for n, p in O.items():
    with safe_open(p, "pt") as f:
        o = {k: f.get_tensor(k) for k in f.keys() if k.startswith(pre)}
    rows = []
    for k, t in sorted(o.items()):
        leaf = k[len(pre):]
        head, _, wb = leaf.rpartition(".")
        kk = pre + MAP.get(head, head) + "." + wb
        u = s[kk]
        same = torch.equal(t.reshape(u.shape).to(u.dtype), u)
        rows.append(dict(legacy=k, svd=kk, shape=list(u.shape), bit_identical=bool(same),
                         max_abs_diff=float((t.reshape(u.shape).double() - u.double()).abs().max())))
    res[n] = dict(n=len(rows), n_identical=sum(r["bit_identical"] for r in rows), rows=rows)
    print(f"{n}: {res[n]['n_identical']}/{res[n]['n']} attention tensors bit-identical to SVD's after name mapping; "
          f"max|d| {max(r['max_abs_diff'] for r in rows):.3e}")
json.dump(res, open(OUTJ, "w"), indent=1)
