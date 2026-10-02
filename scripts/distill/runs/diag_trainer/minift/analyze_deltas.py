"""Relative weight change and direction cosines of the 15 trained up_blocks.3 attn1 tensors (step300.pt of each run dir) vs origin,
and against reference runs (default: the all-sigma P1 runs minift/null and minift/pos).
usage: python analyze_deltas.py <run_dir> [<run_dir> ...]   (env REFS=null,pos to change the reference runs)"""
import os, sys, glob, torch, torch.nn.functional as F
from safetensors import safe_open
M = "/home/kawa/master_project/StereoCrafter/scripts/distill/runs/diag_trainer/minift"
orig = {}
with safe_open(glob.glob("/home/kawa/master_project/StereoCrafter/weights/StereoCrafter/unet_diffusers/*.safetensors")[0], "pt", device="cpu") as sf:
    for k in sf.keys():
        if k.startswith("up_blocks.3.attentions.") and ".transformer_blocks.0.attn1." in k: orig[k] = sf.get_tensor(k).float()
KEYS = sorted(orig); W = torch.cat([orig[k].flatten() for k in KEYS])
def delta(d, step=300):
    sd = torch.load(os.path.join(d, f"step{step}.pt"), map_location="cpu"); return torch.cat([(sd[k].float() - orig[k]).flatten() for k in KEYS])
refs = {r: delta(os.path.join(M, r)) for r in os.environ.get("REFS", "null,pos").split(",")}
print(f"{'run':16s} {'rel||dW||/||W||':>16s} " + " ".join(f"{'cos_vs_'+r:>14s}" for r in refs))
for r, dr in refs.items(): print(f"{r:16s} {float(dr.norm()/W.norm()):16.5f} " + " ".join(f"{float(F.cosine_similarity(dr, d2, dim=0)):14.3f}" for d2 in refs.values()))
for d in sys.argv[1:]:
    dd = delta(d); name = os.path.basename(d.rstrip("/"))
    print(f"{name:16s} {float(dd.norm()/W.norm()):16.5f} " + " ".join(f"{float(F.cosine_similarity(dd, dr, dim=0)):14.3f}" for dr in refs.values()))
