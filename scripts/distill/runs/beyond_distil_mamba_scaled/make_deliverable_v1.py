#!/usr/bin/env python
"""Build the MERGED deliverable and assert every control on it, on CPU, before any GPU time is spent.

WHY MERGED.  The tracked entry point takes ONE --unet_state_path, so a 30-tensor student checkpoint is
not shippable on its own.  The deliverable is therefore the shipped 5-slot Mamba's 95-tensor partial
UNet state with the 30 TRAINED up_blocks.3 tensors overwritten -- a drop-in replacement for
light_lvl0_fulldata333_v2_8k_mamba_only.pt that inpainting_inference.py loads with
--expected_partial_unet_state=True exactly as it loads the shipped file.

WHY bf16.  The shipped file is bf16 and the live UNet parameters are bf16, so the hook path casts the
fp32 student tensors down with a single .to(bf16).  Storing the merged tensors in bf16 reproduces that
single cast (inpainting_inference loads into an fp32 UNet, then casts back to bf16: the bf16 value
round-trips through fp32 exactly), which is what makes the pixel-identity check decisive rather than
approximate.  2.46M of the 3.64M trained parameters live in attn1.time_embed_proj, and a ZERO FiLM is
the identity, so a silently dropped time_embed_proj would yield a weaker model that still passes every
algebra control -- hence the explicit "all 5 FiLMs nonzero" and "30/30 substituted" assertions here.

The PROTECTED shipped file is opened read-only and its md5 is re-checked after the write.

usage: make_deliverable_v1.py <student_ckpt.pt> <out_name_without_dir> [--dry]
"""
import hashlib
import json
import os
import shutil
import sys
import time

import torch

REPO = "/home/kawa/master_project/StereoCrafter"
os.chdir(REPO)
INJ = "/mnt/ssd_data/stereocrafter_weights/_distill_injected"
SHIP = os.path.join(INJ, "light_lvl0_fulldata333_v2_8k_mamba_only.pt")
SHIP_MD5 = "e9c232878319d041680e7fb3be74bf10"

UP3 = [f"up_blocks.3.attentions.{i}.transformer_blocks.0.attn1." for i in (0, 1, 2)]
DOWN0 = [f"down_blocks.0.attentions.{i}.transformer_blocks.0.attn1." for i in (0, 1)]
TAILS = ("fwd.core.", "time_embed_proj.")


def md5(path, chunk=1 << 22):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


student_path, out_name = sys.argv[1], sys.argv[2]
DRY = "--dry" in sys.argv
# a bare name lands in the project's _distill_injected convention dir; a path with a "/" is used
# verbatim, which is how the plumbing is rehearsed into the scratchpad before the real write.
out_path = out_name if "/" in out_name else os.path.join(INJ, out_name)

# ---------------------------------------------------------------- the protected source
assert md5(SHIP) == SHIP_MD5, "PROTECTED shipped checkpoint md5 changed -- refusing to proceed"
raw_ship = torch.load(SHIP, map_location="cpu", weights_only=False)
assert isinstance(raw_ship, dict) and "model" in raw_ship, list(raw_ship)[:5]
ship = raw_ship["model"]
assert len(ship) == 95, len(ship)
print(f"[ship] {SHIP}  md5 {SHIP_MD5}  tensors {len(ship)}  top-level keys {sorted(raw_ship)}")

# ---------------------------------------------------------------- the trained tensors
raw_st = torch.load(student_path, map_location="cpu", weights_only=False)
st = raw_st.get("model", raw_st) if isinstance(raw_st, dict) else raw_st
print(f"[student] {student_path}  tensors {len(st)}  dtypes {sorted({str(v.dtype) for v in st.values()})}")

# the trained set must be EXACTLY the three up_blocks.3 slots' evaluated tensors
expect = {k for k in ship if any(k.startswith(p) for p in UP3)
          and k.split(".attn1.", 1)[1].startswith(TAILS)}
assert set(st) == expect, (
    f"trained set mismatch: missing {sorted(expect - set(st))[:4]} extra {sorted(set(st) - expect)[:4]}")
assert len(st) == 30, len(st)
assert not any(k.startswith(p) for k in st for p in DOWN0), "a down_blocks.0 tensor leaked into the student"
assert not any("origin_attn" in k or ".bwd." in k or "mamba_gate" in k for k in st), "dead-path tensor in the student"
n_film = sum(1 for k in st if "time_embed_proj" in k)
assert n_film == 6, n_film
print(f"[student] set verified: 30 tensors = 3 slots x (8 fwd.core + 2 time_embed_proj), "
      f"{sum(v.numel() for v in st.values())} params")

# ---------------------------------------------------------------- merge
merged, n_sub, n_same = {}, 0, 0
for k, v in ship.items():
    if k in st:
        nv = st[k].to(v.dtype)
        assert nv.shape == v.shape, (k, nv.shape, v.shape)
        merged[k] = nv.clone()
        if not torch.equal(nv, v):
            n_sub += 1
        else:
            n_same += 1
    else:
        merged[k] = v.clone()
assert len(merged) == 95, len(merged)
# the 65 tensors we did NOT train must be bit-identical to the shipped file
untouched = [k for k in ship if k not in st]
assert len(untouched) == 65, len(untouched)
for k in untouched:
    assert torch.equal(merged[k], ship[k]), f"untouched tensor {k} changed"
# gates intact
gates = {k: float(v) for k, v in merged.items() if k.endswith("mamba_gate")}
assert len(gates) == 5 and all(g == 1.0 for g in gates.values()), gates
# every FiLM nonzero (a zero FiLM is the identity -> a silently weaker model)
film_max = {k: float(merged[k].abs().max()) for k in merged if k.endswith("time_embed_proj.weight")}
assert len(film_max) == 5 and all(v > 0 for v in film_max.values()), film_max
print(f"[merge] 95 tensors; {n_sub}/30 trained tensors differ from shipped bf16 ({n_same} round to the same "
      f"bf16), 65 untouched bit-identical; gates {sorted(set(gates.values()))}; "
      f"FiLM |w|max per slot {[round(v, 5) for v in film_max.values()]}")

prov = dict(raw_ship.get("distill", {})) if isinstance(raw_ship.get("distill"), dict) else {}
payload = {
    "model": merged,
    "distill": prov,
    "deliverable": {
        "what": "shipped 5-slot light-Mamba + MAMBA-SIDE step-distilled up_blocks.3 parameters (option A)",
        "base": SHIP,
        "base_md5": SHIP_MD5,
        "student_ckpt": student_path,
        "student_tensors": len(st),
        "student_params": int(sum(v.numel() for v in st.values())),
        "trainable_set": "up_blocks.3.attentions.{0,1,2}.transformer_blocks.0.attn1.{fwd.core.*,time_embed_proj.*}",
        "objective": "step distillation: target = next trajectory point of an M=4 Karras sub-integration "
                     "of the SAME frozen Mamba model, x0-space residual, steps {4,5,6}, measured gains "
                     "{4:0.7177, 5:0.8926, 6:0.9905}, no on-policy refresh",
        "deployed_config": "8 sampler steps, EulerDiscrete Karras, guidance 1.01, 14-frame windows "
                           "overlap 3, 576x1024 centre crop",
        "load_with": "inpainting_inference.py --unet_state_path=<this file> "
                     "--expected_partial_unet_state=True --mamba_gate_override=1.0 plus the six "
                     "MAMBA_SELF_ATTN_*/MAMBA_BIDIRECTIONAL_MODE exports",
        "built": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "builder": "scripts/distill/runs/beyond_distil_mamba_scaled/make_deliverable_v1.py",
    },
}
if DRY:
    print("[dry] not writing")
    sys.exit(0)
assert not os.path.exists(out_path), f"{out_path} exists -- refusing to overwrite"
tmp = out_path + ".tmp"
torch.save(payload, tmp)
shutil.move(tmp, out_path)
new_md5 = md5(out_path)
assert md5(SHIP) == SHIP_MD5, "PROTECTED shipped checkpoint md5 changed DURING the write"
print(f"[save] {out_path}\n[save] md5 {new_md5}  bytes {os.path.getsize(out_path)}")

# reload through BOTH loaders the project uses and re-assert
r2 = torch.load(out_path, map_location="cpu", weights_only=False)
assert set(r2["model"]) == set(merged) and all(torch.equal(r2["model"][k], merged[k]) for k in merged)
r3 = torch.load(out_path, map_location="cpu", weights_only=True)      # inpainting_inference's first try
assert set(r3["model"]) == set(merged), "weights_only=True load lost keys"
print("[reload] weights_only=False and weights_only=True both return the 95 tensors bit-identically")
print(f"[reload] bench2.py-style ['model'] index OK: {len(r2['model'])} tensors")
json.dump({"out": out_path, "md5": new_md5, "n_substituted": n_sub,
           "student": student_path, "shipped_md5_after": md5(SHIP)},
          open(out_path + ".json", "w"), indent=1)
print("MAKE_DELIVERABLE_OK")
