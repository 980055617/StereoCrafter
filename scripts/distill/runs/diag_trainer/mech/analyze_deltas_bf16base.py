"""analyze_deltas.py, corrected for the dtype the training actually started from.

xcheck_mini_ft.py / xcheck_traj_ft.py load the UNet with torch_dtype=bfloat16 and initialise the fp32 AdamW masters
from the *bf16* parameters.  Comparing the saved fp32 masters against the fp32/fp16 safetensors therefore adds a
constant bf16-rounding vector that is IDENTICAL for every run, inflating both ||dW||/||W|| and every pairwise cosine.
This script reports both baselines so the contamination is visible.
usage: python analyze_deltas_bf16base.py <run_dir> [<run_dir> ...]
"""
import os, sys, glob, torch, torch.nn.functional as F
from safetensors import safe_open
R = "/home/kawa/master_project/StereoCrafter"
MF = f"{R}/scripts/distill/runs/diag_trainer/minift"
orig = {}
with safe_open(glob.glob(f"{R}/weights/StereoCrafter/unet_diffusers/*.safetensors")[0], "pt", device="cpu") as sf:
    for k in sf.keys():
        if k.startswith("up_blocks.3.attentions.") and ".transformer_blocks.0.attn1." in k:
            orig[k] = sf.get_tensor(k)
KEYS = sorted(orig)
print("origin safetensors dtype:", {str(orig[k].dtype) for k in KEYS})
Wf = torch.cat([orig[k].float().flatten() for k in KEYS])
Wb = torch.cat([orig[k].to(torch.bfloat16).float().flatten() for k in KEYS])
cast = Wb - Wf
print(f"bf16 cast of the 15 tensors: ||W_bf16 - W_fp||/||W|| = {float(cast.norm()/Wf.norm()):.6f}  "
      f"(this is the floor every run's 'weight change' inherits when measured against the fp checkpoint)")

RUNS = [("null", f"{MF}/null"), ("pos", f"{MF}/pos"), ("null_hi5", f"{MF}/null_hi5"), ("null_mid1", f"{MF}/null_mid1"),
        ("null_low2", f"{MF}/null_low2"), ("pos_hi5", f"{MF}/pos_hi5"), ("pos_lognormal", f"{MF}/pos_lognormal")]
RUNS += [(os.path.basename(d.rstrip('/')), d) for d in sys.argv[1:]]
D = {}
for name, d in RUNS:
    f = os.path.join(d, "step300.pt")
    if not os.path.exists(f):
        print(f"[skip] {name}: no step300.pt"); continue
    sd = torch.load(f, map_location="cpu", weights_only=False)
    D[name] = torch.cat([sd[k].float().flatten() for k in KEYS])

print(f"\n{'run':14s} {'vs fp32 origin':>15s} {'vs bf16 origin':>15s} {'true/reported':>14s}")
for n, v in D.items():
    a = float((v - Wf).norm() / Wf.norm()); b = float((v - Wb).norm() / Wb.norm())
    print(f"{n:14s} {a:15.6f} {b:15.6f} {b/a:14.3f}")

names = list(D)
print(f"\npairwise cosine of the TRUE training deltas (w.r.t. the bf16 starting point):")
print(f"{'':14s} " + " ".join(f"{n[:13]:>13s}" for n in names))
for n in names:
    dn = D[n] - Wb
    print(f"{n:14s} " + " ".join(f"{float(F.cosine_similarity(dn, D[m]-Wb, dim=0)):13.3f}" for m in names))
print(f"\nfor comparison, the same cosines measured against the fp32 checkpoint (what analyze_deltas.py printed):")
print(f"{'':14s} " + " ".join(f"{n[:13]:>13s}" for n in names))
for n in names:
    dn = D[n] - Wf
    print(f"{n:14s} " + " ".join(f"{float(F.cosine_similarity(dn, D[m]-Wf, dim=0)):13.3f}" for m in names))
