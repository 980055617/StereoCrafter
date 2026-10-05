"""deep_20261004 / sdedit lane: hole fill of the SDEdit init SOURCE image (variant F).

fill_all_rowlin(img_hwc, holes_hw): row-wise linear interpolation over EVERY maximal horizontal run of hole pixels
(any length), between the nearest non-hole pixel to the left (s-1) and to the right (e) of the run [s, e); at a region
border the one existing neighbour is copied (crackfill_COPY.fill_rowlin, the stripes lane's 'rowlin', unchanged).
A run that spans the whole row has no neighbour and is left untouched (counted).
hole pixel = mask >= 0.5 (the pipeline's mask_processor binarisation), decided by the caller.
Returns (filled copy, stats); asserts that no non-hole pixel changed.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import crackfill_COPY as _CF  # noqa: E402


def hole_runs(holes_hw):
    h, w = holes_hw.shape
    pad = np.zeros((h, 1), np.int8)
    d = np.diff(np.concatenate([pad, holes_hw.astype(np.int8), pad], axis=1), axis=1)
    rs, cs = np.nonzero(d == 1)
    re_, ce = np.nonzero(d == -1)
    assert len(rs) == len(re_) and np.array_equal(rs, re_), "run pairing failed"
    return rs, cs, ce


def fill_all_rowlin(img_hwc, holes_hw):
    h, w = holes_hw.shape
    assert img_hwc.shape[:2] == (h, w), (img_hwc.shape, holes_hw.shape)
    rs, cs, ce = hole_runs(holes_hw)
    full = (cs == 0) & (ce == w)
    keep = ~full
    out = np.array(img_hwc, copy=True)
    _CF.fill_rowlin(out, (rs[keep], cs[keep], ce[keep]), w)
    chg = np.any(out != img_hwc, axis=-1)
    assert not (chg & ~holes_hw).any(), "fill touched a non-hole pixel"
    return out, dict(runs=int(len(rs)), full_rows=int(full.sum()), lens=ce - cs, changed=chg)
