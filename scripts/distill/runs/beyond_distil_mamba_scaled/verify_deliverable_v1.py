#!/usr/bin/env python
"""Prove the MERGED deliverable is the same model as the hook path, parameter by parameter, on CPU.

Builds the deployed Mamba UNet twice with mmlib.build_pipe_mamba (which mirrors
inpainting_inference.main's construction and state load exactly):
    A) ck = the merged deliverable                      -- the tracked --unet_state_path route
    B) ck = the shipped Mamba, then mmlib.load_swap_mamba(student)  -- the hook route used for the ladder
and asserts EVERY named parameter and buffer of the two UNets is bit-identical.  That is strictly
stronger than a pixel check on one clip and costs no GPU, so it runs first; the pixel check on a real
clip then confirms it end to end through the tracked entry point.

Also re-asserts, on the merged file: 5 gated modules, gate 1.0 with reference_disabled, 95/95
checkpoint tensors present and bit-exact after the fp32 -> bf16 round trip, 0 unexpected keys, and all
5 time_embed_proj FiLMs present and nonzero (a zero FiLM is the identity, so a silently dropped FiLM
would be invisible to every algebra control).

usage: CUDA_VISIBLE_DEVICES='' verify_deliverable_v1.py <merged.pt> <student.pt>
"""
import os
import sys

REPO = "/home/kawa/master_project/StereoCrafter"
sys.path.insert(0, os.path.join(REPO, "scripts/distill/runs/beyond_distil_mamba"))
sys.path.insert(0, REPO)
os.chdir(REPO)

import mmlib as MM                                      # noqa: E402  publishes the deployed Mamba env
import torch                                            # noqa: E402

MERGED, STUDENT = sys.argv[1], sys.argv[2]
DEV = os.environ.get("VD_DEV", "cpu")

print(f"=== A) merged deliverable through the deployed construction: {MERGED}", flush=True)
pA, iA = MM.build_pipe_mamba(dt=torch.bfloat16, dev=DEV, ck=MERGED, gate=1.0)
assert iA["n_gated"] == 5, iA
assert iA["ck_tensors"] == 95, iA
assert iA["ck_absent"] == [] and iA["ck_mismatch"] == [], iA
assert iA["n_unexpected"] == 0, iA
assert iA["materialized"] == 5, iA
assert all(g == 1.0 and rd for _, g, rd in iA["gates"]), iA["gates"]
assert all(present and wmax > 0 for present, wmax in iA["film"].values()), iA["film"]
print(f"[A] gated 5, gate 1.0 + reference_disabled on all 5, materialized 5, 95/95 tensors bit-exact "
      f"after the fp32->bf16 round trip, unexpected 0, all 5 FiLMs nonzero "
      f"{[round(v[1], 5) for v in iA['film'].values()]}", flush=True)

print(f"\n=== B) shipped Mamba + the hook swap of {STUDENT}", flush=True)
pB, iB = MM.build_pipe_mamba(dt=torch.bfloat16, dev=DEV, ck=MM.MAMBA_CK, gate=1.0)
assert iB["n_gated"] == 5 and iB["ck_absent"] == [] and iB["ck_mismatch"] == [], iB
nsel, ndiff = MM.load_swap_mamba(pB.unet, STUDENT)
print(f"[B] swapped {nsel} tensors, {ndiff} differed from the shipped Mamba", flush=True)

print("\n=== full-parameter comparison A vs B", flush=True)
PA = dict(pA.unet.named_parameters()); PB = dict(pB.unet.named_parameters())
BA = dict(pA.unet.named_buffers()); BB = dict(pB.unet.named_buffers())
assert set(PA) == set(PB) and set(BA) == set(BB), "module structure differs"
bad = [k for k in PA if not torch.equal(PA[k], PB[k])]
badb = [k for k in BA if not torch.equal(BA[k].float(), BB[k].float())]
print(f"[cmp] parameters {len(PA)} compared, {len(bad)} differ; buffers {len(BA)} compared, {len(badb)} differ")
if bad:
    for k in bad[:6]:
        print(f"  DIFF {k} maxabs {float((PA[k].float()-PB[k].float()).abs().max()):.3e}")
assert not bad and not badb, "MERGED DELIVERABLE IS NOT THE HOOK MODEL"
# and the trained slots really did move away from the shipped weights
ship, _ = torch.load(MM.MAMBA_CK, map_location="cpu", weights_only=False)["model"], None
moved = sum(1 for k in ship if k in PA and not torch.equal(PA[k].detach().cpu(), ship[k].to(PA[k].dtype)))
print(f"[cmp] of the 95 shipped tensors, {moved} differ in the merged model (expected: the trained up3 subset)")
assert moved == ndiff, (moved, ndiff)
print("VERIFY_DELIVERABLE_OK: the merged file and the hook path are the same model, bit for bit")
