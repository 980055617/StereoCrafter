#!/usr/bin/env python
"""vae_20261005 / decoder_swap -- STEP 3 HEADROOM table (PREREG.txt section 2; descriptive, never a selection input).  CPU.
Source: outputs/vae_20261005/decoder_swap/headroom_dev_v1/<clip>.json (score_headroom_v1.py).
usage: python analyze_headroom_v1.py  (prints; the caller redirects to TABLE_HEADROOM_v1.txt)
"""
import json
import os

import numpy as np

REPO = "/home/kawa/master_project/StereoCrafter"
os.chdir(REPO)
SRC = "outputs/vae_20261005/decoder_swap/headroom_dev_v1"
DEV = "0040 0082 0091 0184 0245 0268".split()
ROWS = ["stock", "stock32", "ftmse", "ftema", "cd", "sd15"]
J = {c: json.load(open(f"{SRC}/{c}.json")) for c in DEV}
print(f"STEP 3 HEADROOM (dev, 6 clips): decoder round trip of the REAL right eye (deployed bf16 encode, latents shared by all "
      f"decoders) scored against the encoded frames themselves (registration-free), frames 0,4,8,...  source {SRC}/<clip>.json")
print("HR0 reproduction gate (stock row vs decoder_ft ROUNDTRIP_DEV_main_v1.json):")
for c in DEV:
    h = J[c]["rows"]["stock"]["HR0"]
    print(f"  {c}: md5 {'EQUAL' if h['md5_equal'] else 'DIFFERENT'}; LPIPS {J[c]['rows']['stock']['lpips']:.6f} vs "
          f"{h['decoder_ft_lpips']:.6f} (d {h['lpips_diff']:+.1e}) -> {'PASS' if (h['md5_equal'] or abs(h['lpips_diff']) <= 1e-4) else 'FAIL'}")
print(f"\n{'decoder':8s} {'LPIPS':>7s} {'PSNR':>7s} {'DISTS':>7s} {'NIQE':>6s} {'MUSIQ':>6s} {'edgeHF/GT':>9s} {'flatHF/GT':>9s} "
      f"{'stripe/GT':>9s} {'halo%':>6s}  LPIPS better than stock (of 6)  s/frame")
for r in ROWS:
    v = [J[c]["rows"][r] for c in DEV]
    nb = sum(J[c]["rows"][r]["lpips"] < J[c]["rows"]["stock"]["lpips"] for c in DEV)
    print(f"{r:8s} {np.mean([x['lpips'] for x in v]):7.4f} {np.mean([x['psnr'] for x in v]):7.3f} {np.mean([x['dists'] for x in v]):7.4f} "
          f"{np.mean([x['niqe'] for x in v]):6.3f} {np.mean([x['musiq'] for x in v]):6.2f} "
          f"{np.nanmean([x['ratio_to_GT']['edgeHF'] for x in v]):9.3f} {np.nanmean([x['ratio_to_GT']['flatHF'] for x in v]):9.3f} "
          f"{np.nanmean([x['ratio_to_GT']['stripeE'] for x in v]):9.3f} {np.mean([x['decompose']['haloFrac'] for x in v]):6.3f}  "
          f"{nb}/6{'':27s}{np.mean([x['decode']['sec_per_decoded_frame'] for x in v]):.3f}")
print(f"{'GT':8s} {'':7s} {'':7s} {'':7s} {np.mean([J[c]['GT']['niqe'] for c in DEV]):6.3f} {np.mean([J[c]['GT']['musiq'] for c in DEV]):6.2f} "
      f"{1.0:9.3f} {1.0:9.3f} {1.0:9.3f} {np.mean([J[c]['GT']['decompose']['haloFrac'] for c in DEV]):6.3f}")
print("  (/GT ratios: nanmean over clips; a clip whose GT region value is exactly 0 gives NaN and is left out -- "
      + ", ".join(f"{c}: " + "/".join(k for k in ("edgeHF", "flatHF", "stripeE") if J[c]["GT"]["decompose"][k] == 0) for c in DEV
                  if any(J[c]["GT"]["decompose"][k] == 0 for k in ("edgeHF", "flatHF", "stripeE"))) + ")")
print("\nper-clip LPIPS-Alex (lower = closer to the real right eye):")
print(f"  {'clip':6s} " + " ".join(f"{r:>8s}" for r in ROWS))
for c in DEV:
    print(f"  {c:6s} " + " ".join(f"{J[c]['rows'][r]['lpips']:8.4f}" for r in ROWS))
print("\nper-clip PSNR (dB):")
for c in DEV:
    print(f"  {c:6s} " + " ".join(f"{J[c]['rows'][r]['psnr']:8.3f}" for r in ROWS))
print("\nHR-SEED (consistency decoder noise dependence on the REAL right eye's latents; mean |a-b| RGB in [0,1], frames 0,4,8,..):")
for c in DEV:
    hs = J[c].get("HR_SEED")
    if not hs:
        continue
    for k, v in hs.items():
        if v:
            print(f"  {c} {k:16s} all {v['all']:.5f}  GT-edge {v['edge']:.5f}  GT-flat {v['flat']:.5f}")
