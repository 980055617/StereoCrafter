#!/bin/bash
# Smoke tests S1-S6 for the manifest capture (GPU 0). Writes the temb reference on the first run.
set -u; cd /home/kawa/master_project/StereoCrafter; D=scripts/distill; PY=/home/kawa/miniconda3/envs/stereocrafter/bin/python
BEST=/mnt/ssd_data/stereocrafter_weights/_distill_injected/light_lvl0_multiclip13_r2_mamba_only.pt
E210=/mnt/ssd_data/stereocrafter_weights/Overfit0160_LightMamba_Lvl0_ds128_fwd_FromE140/MambaCrafter_20260912_044603/train_state_epoch000210.pt
M=$D/runs/fulldata/manifests; S=/mnt/ssd_data/attn_cache/smoke; rm -rf $S; mkdir -p $S; export CUDA_VISIBLE_DEVICES=0
F='\[capture\]|REFUSED|ABORT|Traceback|Error|assert'
echo "== S3/S4/S1: 3-window manifest (2160 / 4400 / deep seek), writes temb ref =="; rm -f $D/runs/fulldata/temb_ref_origin.pt
CAP_MANIFEST=$M/smoke3.json CAP_OUT=$S/tf3 CKPT=$BEST $PY $D/capture_fulldata.py 2>&1 | grep -E "$F"
echo "== S6: stubs off, same manifest -> x must be bit-identical =="
CAP_MANIFEST=$M/smoke3.json CAP_OUT=$S/tf3_nostub CKPT=$BEST CAP_NO_STUB=1 $PY $D/capture_fulldata.py 2>&1 | grep -E "$F"
$PY - <<'PY'
import torch, glob, os
a=sorted(glob.glob('/mnt/ssd_data/attn_cache/smoke/tf3/*/*.pt')); n=0; bad=0
for f in a:
    g=f.replace('/tf3/','/tf3_nostub/'); 
    if not os.path.exists(g): bad+=1; continue
    x=torch.load(f)['x']; y=torch.load(g)['x']; n+=1; bad+= int(not torch.equal(x,y))
print(f"S6 stub check: {n} files compared, mismatches={bad}", "PASS" if bad==0 and n>0 else "FAIL")
PY
echo "== S3b: 0160 w0 all 28 rows vs LEGACY cache (torch.manual_seed(1234) once == per-window seed 1234 for idx 0) =="
CAP_MANIFEST=$M/smoke1.json CAP_OUT=$S/tf1_all CKPT=$BEST CAP_STRAT=0 CAP_KEEP=28 $PY $D/capture_fulldata.py 2>&1 | grep -E "$F"
$PY - <<'PY'
import torch, glob
leg=torch.load('/mnt/ssd_data/attn_cache/0160_origin_tf/down_blocks.0.attentions.0.transformer_blocks.0.attn1__call0000.pt')
idx=leg['idx']; X=leg['x']; ok=0; tot=0; maxd=0.0
files={ torch.load(f)['meta']['row']: torch.load(f)['x'] for f in glob.glob('/mnt/ssd_data/attn_cache/smoke/tf1_all/d0a0/0160_w00000_s0_*.pt') }
for k,i in enumerate(idx):
    if i in files:
        tot+=1; d=(files[i].float()-X[k].float()).abs().max().item(); maxd=max(maxd,d); ok+=int(d==0.0)
print(f"S3b legacy-vs-manifest x (d0a0, window0, step0): rows compared={tot} identical={ok} max|diff|={maxd:.3e}", "PASS" if tot==4 and ok==4 else ("NEAR" if maxd<1e-2 else "FAIL"))
PY
echo "== S4 negative control: FULL e210 checkpoint must be REFUSED, and with CAP_ALLOW_FULL=1 the temb guard must ABORT =="
CAP_MANIFEST=$M/smoke1.json CAP_OUT=$S/neg CKPT=$E210 $PY $D/capture_fulldata.py 2>&1 | grep -E "$F" | head -2
CAP_MANIFEST=$M/smoke1.json CAP_OUT=$S/neg2 CKPT=$E210 CAP_ALLOW_FULL=1 $PY $D/capture_fulldata.py 2>&1 | grep -E "$F" | tail -3
echo "== sizes =="; du -sh $S/*; ls $S/tf3/d0a0 | head -4; ls $S/tf3/d0a0 | wc -l; cat $S/tf3/capture_meta_*.json | python3 -c "import json,sys; m=json.load(sys.stdin); print({k:m[k] for k in ('windows','calls','files','s_per_window','temb_ok','peak_rss_gb','mask_frac')})"
echo SMOKE_DONE
