"""CLAIM1 of scripts/distill/runs/fulldata_v2/beyond4/verify_lossless.py for many files: the decoded FFV1 array's md5
equals the pre-encode md5 the writer recorded in <file>.mkv.md5.  CPU only.  usage: python verify_ll_files.py f1.mkv ..."""
import hashlib, sys
import numpy as np
from decord import VideoReader, cpu
ok = True
for p in sys.argv[1:]:
    vr = VideoReader(p, ctx=cpu(0)); A = vr.get_batch(list(range(len(vr)))).asnumpy()
    pre = open(p + ".md5").read().split()[0]
    got = hashlib.md5(np.ascontiguousarray(A).tobytes()).hexdigest()
    ok &= (pre == got)
    print(f"{'MATCH' if pre == got else 'MISMATCH'} shape={A.shape} pre={pre} decoded={got} {p}", flush=True)
    del A, vr
print("ALL_LOSSLESS" if ok else "NOT_ALL_LOSSLESS")
