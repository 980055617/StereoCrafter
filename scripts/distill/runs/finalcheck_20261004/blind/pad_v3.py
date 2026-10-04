#!/usr/bin/env python
"""items_v3 (pad_v3.py = pad_v2.py with one flag renamed: ab_jpeg_bytes_equal_per_item -> ab_sizes_equal_per_item,
which had tripped its own leak-check pattern 'jpeg_bytes' as a false positive) = items_v1 with two blinding leaks closed; PIXELS UNCHANGED, windows/frames/order/key frozen.

Leak 1 (JPEG size): the deliverable panel's JPEG was larger than origin's on 36/36 items, so base64 lengths in
  items.json / file sizes in panels/ unblind.  Fix: per item, pad BOTH A and B to max(lenA,lenB)+4 bytes with
  JPEG COM segments (FF FE + 2-byte length + zero payload) inserted right after SOI.  Every padded JPEG must
  decode np.array_equal to its v1 panel.  (Residual: a deliberate JPEG-segment parse still shows the padding.)
Leak 2 (per-panel stats outside _KEY files): v1's build.log prints 'std GT/O/D' per clip/slot and v1's
  checks.json carries per-item std_min (often = origin's std).  Fix: v2's non-_KEY files carry only
  model-independent (GT / geometry / mask / render-LEFT, which is bit-identical for A and B) or A/B-symmetric
  per-item data; everything else goes to _KEY_* files.  build.log is NOT copied.

usage: python pad_v2.py V1DIR OUTDIR        (OUTDIR must not exist; CPU only)
"""
import base64
import io
import json
import os
import shutil
import struct
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blindlib as B  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

V1, OUT = sys.argv[1], sys.argv[2]
B.new_dir(OUT)
os.makedirs(f"{OUT}/panels")
log = B.Tee(f"{OUT}/pad.log")
MAX_ITEMS_BYTES = 12 * 1024 * 1024


def com_pad(jpg: bytes, n: int) -> bytes:
    """Insert COM segments adding exactly n (>= 4) bytes after SOI (+ the JFIF APP0 segment, if present,
    so the file stays JFIF-conformant: APP0 must immediately follow SOI)."""
    assert jpg[:2] == b"\xff\xd8", "not a JPEG (no SOI)"
    assert n >= 4
    pos = 2
    if jpg[2:4] == b"\xff\xe0":                # APP0 / JFIF
        pos = 4 + struct.unpack(">H", jpg[4:6])[0]
    segs, left = [], n
    while left > 0:
        add = min(left, 65537)                 # one COM segment adds L+2 bytes, L in [2, 65535]
        if 0 < left - add < 4:                 # never leave a remainder too small for its own segment
            add = left - 4
        L = add - 2
        segs.append(b"\xff\xfe" + struct.pack(">H", L) + b"\x00" * (L - 2))
        left -= add
    out = jpg[:pos] + b"".join(segs) + jpg[pos:]
    assert len(out) == len(jpg) + n
    return out


def dec(b):
    return np.array(Image.open(io.BytesIO(b)).convert("RGB"))


items1 = json.load(open(f"{V1}/items.json"))
key1 = json.load(open(f"{V1}/_KEY_do_not_open_before_rating.json"))
meta1 = json.load(open(f"{V1}/items_meta.json"))
checks1 = json.load(open(f"{V1}/checks.json"))
diag1 = json.load(open(f"{V1}/_KEY_panel_diagnostics_do_not_open_before_rating.json"))

items2, diag2 = [], {}
for it in items1:
    gt = base64.b64decode(it["gt_b64"])
    a, b = base64.b64decode(it["a_b64"]), base64.b64decode(it["b_b64"])
    target = max(len(a), len(b)) + 4
    a2, b2 = com_pad(a, target - len(a)), com_pad(b, target - len(b))
    assert len(a2) == len(b2) == target
    assert np.array_equal(dec(a2), dec(a)) and np.array_equal(dec(b2), dec(b)), it["id"]
    # the padded panels must also equal the v1 files on disk pixel for pixel
    for lab, raw in (("A", a), ("B", b), ("GT", gt)):
        assert open(f"{V1}/panels/{it['id']}_{lab}.jpg", "rb").read() == raw
    it2 = dict(it)
    it2["a_b64"] = base64.b64encode(a2).decode("ascii")
    it2["b_b64"] = base64.b64encode(b2).decode("ascii")
    assert len(it2["a_b64"]) == len(it2["b_b64"])
    items2.append(it2)
    for lab, raw in (("GT", gt), ("A", a2), ("B", b2)):
        with open(f"{OUT}/panels/{it['id']}_{lab}.jpg", "wb") as fh:
            fh.write(raw)
    d = dict(diag1[it["id"]])
    d["jpeg_bytes_v1_unpadded"] = d.pop("jpeg_bytes")
    d["jpeg_bytes_v2_padded"] = {"GT": len(gt), "A": len(a2), "B": len(b2)}
    diag2[it["id"]] = d
    log(f"{it['id']}: A/B padded to an equal {target} bytes; decoded pixels identical to v1")

assert [i["id"] for i in items2] == [i["id"] for i in items1]
assert all(sorted(i.keys()) == sorted(["id", "clip", "frame", "region_type", "window", "gt_b64", "a_b64", "b_b64"])
           for i in items2)
with open(f"{OUT}/items.json", "w") as fh:
    json.dump(items2, fh)
size = os.path.getsize(f"{OUT}/items.json")
B.jdump(key1, f"{OUT}/_KEY_do_not_open_before_rating.json")
assert json.load(open(f"{OUT}/_KEY_do_not_open_before_rating.json")) == key1
B.jdump(diag2, f"{OUT}/_KEY_panel_diagnostics_do_not_open_before_rating.json")
# the full v1 checks (incl. per-item std_min) are identity-revealing -> key side only
B.jdump(checks1, f"{OUT}/_KEY_checks_full_v1_do_not_open_before_rating.json")
shutil.copyfile(f"{V1}/contact_sheet.png", f"{OUT}/contact_sheet.png")   # same decoded pixels by the asserts above

# ---- sanitized, non-identifying checks ------------------------------------------------------------
c = checks1["checks"]
san = dict(
    note="non-identifying subset of items_v1/checks.json: model-independent or A/B-symmetric per-item data only; "
         "the full version is _KEY_checks_full_v1_do_not_open_before_rating.json",
    summary={k: v for k, v in checks1["summary"].items() if k not in ("V8_min_jpeg_psnr",)},
    V3=c["V3"], V4=c["V4"],
    V5=[{k: v for k, v in r.items()} for r in c["V5"]],                       # GT vs render-LEFT (identical for A/B)
    V6=[dict(clip=r["clip"], slot=r["slot"], pass_=bool(r["std_min"] >= 4 and not r["A_eq_B"])) for r in c["V6"]],
    V7=c["V7"],                                                                # GT vs warped input (model-independent)
    V8=[dict(id=r["id"], clip=r["clip"], slot=r["slot"], rms_AB_lossless=r["rms_AB_lossless"],
             rms_jpeg_err_AB=r["rms_jpeg_err_AB"], ratio=r["ratio"], flag_ratio_lt_1p5=r["flag_ratio_lt_1p5"],
             flag_any_panel_psnr_lt_35=r["flag_psnr_lt_35"]) for r in c["V8"]],
    V9=dict(checks1["summary"]["V9"], items_json_bytes=size, under_12MB=size < MAX_ITEMS_BYTES,
            ab_sizes_equal_per_item=True, padding="JPEG COM segments after SOI; pixels unchanged"),
)
B.jdump(san, f"{OUT}/checks.json")

meta2 = json.loads(json.dumps(meta1))
meta2["jpeg"]["ab_size_equalized"] = ("per item, A and B padded with JPEG COM segments to the same byte length "
                                      "(max+4); decoded pixels identical to items_v1")
meta2["derived_from"] = V1
B.jdump(meta2, f"{OUT}/items_meta.json")

# ---- leak check over every non-_KEY file -------------------------------------------------------------
bad = []
pats = ["origin@1.01", "deliverable@1.01", "mstudent2", "origin_ll", "GT/O/D", "std_min", "\"O\"", "\"D\"",
        "jpeg_psnr", "jpeg_bytes"]
for fn in sorted(os.listdir(OUT)):
    p = f"{OUT}/{fn}"
    if fn.startswith("_KEY") or not os.path.isfile(p) or fn.endswith(".png"):
        continue
    s = open(p).read()
    if fn == "items_meta.json":                     # its 'sources' block names both renders (no per-item mapping)
        s = json.dumps(json.load(open(p))["items"])
    for pat in pats:
        if pat in s:
            bad.append(f"{fn}: '{pat}'")
sizes = {(os.path.getsize(f"{OUT}/panels/{i['id']}_A.jpg"), os.path.getsize(f"{OUT}/panels/{i['id']}_B.jpg"))
         for i in items2}
assert all(x == y for x, y in sizes)
log(f"items.json {size} bytes ({size/1e6:.3f} MB, < 12 MB: {size < MAX_ITEMS_BYTES}); "
    f"A/B byte lengths equal on {len(items2)}/{len(items2)} items (b64 and files)")
log(f"leak check over non-_KEY files: {'CLEAN' if not bad else bad}")
if bad:
    raise SystemExit("leak check failed")
log("done")
