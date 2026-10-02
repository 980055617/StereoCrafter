"""Bitwise / relative comparison of a run's saved checkpoints against origin's 15 up_blocks.3 attn1 tensors.
usage: python check_ck_vs_origin.py <run_dir> [<run_dir> ...]"""
import os, sys, glob, torch
from safetensors import safe_open
R = "/home/kawa/master_project/StereoCrafter"
orig = {}
with safe_open(glob.glob(f"{R}/weights/StereoCrafter/unet_diffusers/*.safetensors")[0], "pt", device="cpu") as sf:
    for k in sf.keys():
        if k.startswith("up_blocks.3.attentions.") and ".transformer_blocks.0.attn1." in k:
            orig[k] = sf.get_tensor(k)
KEYS = sorted(orig)
W = torch.cat([orig[k].float().flatten() for k in KEYS])
print(f"{'checkpoint':34s} {'bitwise==origin':>16s} {'#tensors differing':>19s} {'rel||dW||/||W||':>16s} {'max|dW|':>12s}")
for d in sys.argv[1:]:
    for f in sorted(glob.glob(os.path.join(d, "step*.pt"))):
        sd = torch.load(f, map_location="cpu", weights_only=False)
        dv = torch.cat([(sd[k].float() - orig[k].float()).flatten() for k in KEYS])
        nd = sum(int(not torch.equal(sd[k].float(), orig[k].float())) for k in KEYS)
        tag = os.path.relpath(f, R).replace("scripts/distill/runs/diag_trainer/", "")
        print(f"{tag[-34:]:34s} {str(nd == 0):>16s} {nd:>19d} {float(dv.norm()/W.norm()):16.8f} {float(dv.abs().max()):12.3e}")
