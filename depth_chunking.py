"""Pure-numpy pieces of the chunked DepthCrafter path (no torch, no decord).

Extracted from depth_splatting_inference.py on 2026-09-17 so the chunk
schedule, the joint chunk-affine fit, the overlap stitching and the final
normalisation can be unit-tested in a plain numpy environment
(scripts/test_depth_chunk_alignment.py). depth_splatting_inference.py
imports everything from here; the private names it used before are kept as
aliases so nothing else changes.

Fixes carried by this module (flow-audit-2026-09-17.md, section 3.D):

- D3: `depth_chunk_ranges` never emits a trailing chunk that adds no new
  frames; the last chunk is pulled back to full length instead.
- D1: `solve_global_chunk_affine` weights each overlap by its p90-p10 spread,
  falls back to "same transform as the previous chunk" for a flat overlap,
  adds a weak ridge toward (1, 0), and re-solves after clipping so a chunk
  whose scale hit the bounds never leaks its unclipped value downstream.
- D7: `normalize_depth_inplace` uses p0.01/p99.99 (flag) instead of the
  absolute min/max that single outlier pixels set; `save_depth_npz_with_extras`
  writes uint16 fixed point via depth_io and records which percentile was
  used.
"""

from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.bundle_common.depth_io import save_depth_npz  # noqa: E402

# Weight of the rows that tie a chunk to its predecessor when their overlap is
# flat, and of the rows that pin a clipped scale during the re-solve. Data
# rows are weighted by the overlap spread (<= ~1 in DepthCrafter units), so
# these dominate without making the system ill-conditioned in float64.
TIE_WEIGHT = 1.0
PIN_WEIGHT = 1e3
DEFAULT_SCALE_BOUNDS = (0.25, 4.0)


# ---------------------------------------------------------------- D3: schedule


def depth_chunk_ranges(total_frames: int, chunk_size: int, chunk_overlap: int) -> List[Tuple[int, int]]:
    """[start, end) ranges over `total_frames` with stride chunk_size - overlap.

    Unlike `range(0, T, step)`, the final chunk is pulled back to
    `T - chunk_size` (same policy as utils.training_batches.chunk_frame_ranges)
    so it is full length and always adds new frames. `range(0, T, step)` emits
    a trailing chunk of `T mod step` frames whenever that is in [1, overlap]:
    it costs a whole DepthCrafter pass, adds nothing, and its short-context
    estimate then dominates the final frames through the overlap blend.
    """
    if total_frames <= 0:
        return []
    if chunk_size <= 0 or total_frames <= chunk_size:
        return [(0, total_frames)]
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be in [0, chunk_size)")
    step = chunk_size - chunk_overlap
    ranges: List[Tuple[int, int]] = []
    start = 0
    while True:
        end = start + chunk_size
        if end >= total_frames:
            ranges.append((total_frames - chunk_size, total_frames))
            return ranges
        ranges.append((start, end))
        start += step


def pair_overlaps(ranges: Sequence[Tuple[int, int]]) -> List[int]:
    """Frames shared by consecutive ranges (wider than the nominal overlap for
    the pulled-back last chunk)."""
    return [max(0, ranges[i][1] - ranges[i + 1][0]) for i in range(len(ranges) - 1)]


# ---------------------------------------------------------- D1: chunk affine


def _depth_sample_values(values: np.ndarray, max_samples: int = 200_000) -> np.ndarray:
    flat = np.asarray(values).reshape(-1)
    valid = flat[np.isfinite(flat)]
    if valid.size == 0:
        return valid.astype(np.float32, copy=False)
    if valid.size > max_samples:
        step = int(np.ceil(valid.size / float(max_samples)))
        valid = valid[::step]
    return valid.astype(np.float32, copy=False)


def _percentile_stats(values: np.ndarray, percentiles: Sequence[float] = (10, 50, 90)) -> Optional[np.ndarray]:
    samples = _depth_sample_values(values)
    if samples.size < 16:
        return None
    return np.percentile(samples, list(percentiles)).astype(np.float64)


def solve_global_chunk_affine(
    raw_chunks: Sequence[np.ndarray],
    overlaps: Union[int, Sequence[int]],
    *,
    min_spread: float = 0.02,
    ridge: float = 1e-5,
    scale_bounds: Tuple[float, float] = DEFAULT_SCALE_BOUNDS,
    percentiles: Sequence[float] = (10, 50, 90),
    report: Optional[List[Dict[str, float]]] = None,
) -> List[Tuple[float, float]]:
    """Jointly fit a per-chunk (scale, offset) so every pairwise overlap agrees
    at once, instead of chaining each chunk onto the previous one. Chaining
    lets each pairwise fit's error compound into every later chunk; solving
    all overlaps together as one least-squares system distributes the same
    residuals instead of accumulating them across the sequence.

    Chunk 0 is the reference (1, 0). Each overlap contributes one row per
    percentile, weighted by the overlap's p90-p10 spread: a near-constant
    overlap (fade to black, static sky, a cut inside the overlap) gives
    collinear rows that used to make lstsq return an arbitrary min-norm answer
    for that chunk and, through the joint solve, for every chunk after it. An
    overlap whose spread is below `min_spread` is not used at all; the later
    chunk simply inherits the earlier chunk's transform (relative (1, 0)). A
    ridge of weight `ridge` pulls every chunk weakly toward (1, 0) so a barely
    non-flat overlap cannot swing the fit. Scales outside `scale_bounds` are
    pinned at the bound and the system is re-solved so the remaining chunks
    are fitted against the value that will actually be applied.

    `overlaps` is the nominal overlap (int) or one value per consecutive pair
    (see pair_overlaps). `report`, if given, receives one dict per pair.
    """
    n_chunks = len(raw_chunks)
    if n_chunks <= 1:
        return [(1.0, 0.0)] * n_chunks
    if isinstance(overlaps, (int, np.integer)):
        overlaps = [int(overlaps)] * (n_chunks - 1)
    overlaps = list(overlaps)
    if len(overlaps) != n_chunks - 1:
        raise ValueError(f"expected {n_chunks - 1} overlaps, got {len(overlaps)}")

    n_unknowns = 2 * (n_chunks - 1)  # (scale_i, offset_i) for i = 1..n_chunks-1

    def col(idx: int) -> int:
        return 2 * (idx - 1)

    rows: List[List[float]] = []
    rhs: List[float] = []

    def add_row(entries: Dict[int, float], value: float) -> None:
        row = [0.0] * n_unknowns
        for index, coeff in entries.items():
            row[index] += coeff
        rows.append(row)
        rhs.append(value)

    for i in range(n_chunks - 1):
        overlap_n = min(int(overlaps[i]), int(raw_chunks[i].shape[0]), int(raw_chunks[i + 1].shape[0]))
        tail = head = None
        if overlap_n > 0:
            tail = _percentile_stats(raw_chunks[i][-overlap_n:], percentiles)
            head = _percentile_stats(raw_chunks[i + 1][:overlap_n], percentiles)
        if tail is None or head is None:
            spread = 0.0
        else:
            spread = float(min(tail[-1] - tail[0], head[-1] - head[0]))
        usable = spread >= min_spread
        if report is not None:
            report.append({"pair": i, "overlap": overlap_n, "spread": spread, "used": float(usable)})
        if not usable:
            # Flat overlap: no information about the relative transform, so
            # chunk i+1 keeps chunk i's transform (chunk 0's is the constant
            # (1, 0)). Tie rows rather than hard substitution so the same
            # lstsq handles it.
            for k, ref_value in ((0, 1.0), (1, 0.0)):
                if i == 0:
                    add_row({col(i + 1) + k: TIE_WEIGHT}, TIE_WEIGHT * ref_value)
                else:
                    add_row({col(i + 1) + k: TIE_WEIGHT, col(i) + k: -TIE_WEIGHT}, 0.0)
            continue
        w = spread
        for x_i, x_j in zip(tail.tolist(), head.tolist()):
            entries: Dict[int, float] = {}
            b_val = 0.0
            if i == 0:
                b_val -= w * x_i
            else:
                entries[col(i)] = w * x_i
                entries[col(i) + 1] = w
            entries[col(i + 1)] = entries.get(col(i + 1), 0.0) - w * x_j
            entries[col(i + 1) + 1] = entries.get(col(i + 1) + 1, 0.0) - w
            add_row(entries, b_val)

    if ridge > 0:
        r = float(np.sqrt(ridge))
        for i in range(1, n_chunks):
            add_row({col(i): r}, r * 1.0)
            add_row({col(i) + 1: r}, 0.0)

    base_rows = np.asarray(rows, dtype=np.float64)
    base_rhs = np.asarray(rhs, dtype=np.float64)
    lo, hi = float(scale_bounds[0]), float(scale_bounds[1])

    def solve(pinned: Dict[int, float]) -> np.ndarray:
        a = base_rows
        b = base_rhs
        if pinned:
            extra = np.zeros((len(pinned), n_unknowns), dtype=np.float64)
            extra_rhs = np.zeros(len(pinned), dtype=np.float64)
            for k, (idx, value) in enumerate(pinned.items()):
                extra[k, col(idx)] = PIN_WEIGHT
                extra_rhs[k] = PIN_WEIGHT * value
            a = np.concatenate([a, extra], axis=0)
            b = np.concatenate([b, extra_rhs], axis=0)
        solution, *_ = np.linalg.lstsq(a, b, rcond=None)
        return solution

    # Clip-and-re-solve: a scale that lands outside the bounds is pinned at
    # the bound and everything else is fitted again, so the chunks after it
    # are aligned to the value that is actually applied, not the raw one.
    # One violator per pass, earliest first: the transforms chain, so once
    # chunk i is pinned a later chunk that only looked out of bounds because
    # of i usually lands back inside.
    pinned: Dict[int, float] = {}
    solution = solve(pinned)
    for _ in range(n_chunks):
        violator = None
        for i in range(1, n_chunks):
            if i in pinned:
                continue
            scale = float(solution[col(i)])
            if not np.isfinite(scale):
                violator = (i, 1.0)
            elif scale < lo or scale > hi:
                violator = (i, float(np.clip(scale, lo, hi)))
            if violator is not None:
                break
        if violator is None:
            break
        pinned[violator[0]] = violator[1]
        solution = solve(pinned)

    result: List[Tuple[float, float]] = [(1.0, 0.0)]
    for i in range(1, n_chunks):
        scale = float(solution[col(i)])
        offset = float(solution[col(i) + 1])
        if not np.isfinite(scale) or not np.isfinite(offset):
            scale, offset = 1.0, 0.0
        scale = float(np.clip(scale, lo, hi))
        result.append((scale, offset))
    return result


# ------------------------------------------------------------- stitching


def _take_tail(parts: List[np.ndarray], count: int) -> np.ndarray:
    if count <= 0:
        raise ValueError("count must be positive")
    out: List[np.ndarray] = []
    remaining = count
    for part in reversed(parts):
        if remaining <= 0:
            break
        take = min(remaining, part.shape[0])
        out.append(part[-take:])
        remaining -= take
    if remaining != 0:
        raise ValueError("not enough frames in parts")
    out.reverse()
    return np.concatenate(out, axis=0) if len(out) > 1 else out[0].copy()


def _replace_tail(parts: List[np.ndarray], values: np.ndarray) -> None:
    remaining = int(values.shape[0])
    value_end = remaining
    for index in range(len(parts) - 1, -1, -1):
        if remaining <= 0:
            break
        part = parts[index]
        take = min(remaining, part.shape[0])
        value_start = value_end - take
        part[-take:] = values[value_start:value_end]
        value_end = value_start
        remaining -= take
    if remaining != 0:
        raise ValueError("not enough frames in parts")


def _blend_depth_overlap(previous: np.ndarray, current: np.ndarray) -> np.ndarray:
    n = int(min(previous.shape[0], current.shape[0]))
    if n <= 0:
        return previous
    weights = (np.arange(n, dtype=np.float32) + 1.0) / float(n + 1)
    weights = weights[:, None, None]
    return previous[:n] * (1.0 - weights) + current[:n] * weights


def _concatenate_depth_parts(parts: List[np.ndarray]) -> np.ndarray:
    if not parts:
        raise ValueError("no depth parts were produced")
    total = sum(int(part.shape[0]) for part in parts)
    height, width = parts[0].shape[1:3]
    out = np.empty((total, height, width), dtype=np.float32)
    offset = 0
    for idx, part in enumerate(parts):
        n = int(part.shape[0])
        out[offset : offset + n] = part.astype(np.float32, copy=False)
        offset += n
        parts[idx] = None  # type: ignore[assignment]
    return out


def stitch_depth_chunks(
    chunks: List[np.ndarray],
    overlaps: Sequence[int],
    expected_total: Optional[int] = None,
    log=None,
) -> np.ndarray:
    """Blend consecutive (already affine-aligned) chunks over their overlaps
    and concatenate. `chunks` entries are released as they are consumed."""
    parts: List[np.ndarray] = []
    unique_frames = 0
    for chunk_id, chunk_depth in enumerate(chunks):
        overlap_n = int(overlaps[chunk_id - 1]) if chunk_id > 0 else 0
        overlap_n = min(overlap_n, int(chunk_depth.shape[0]), int(unique_frames))
        if parts and overlap_n > 0:
            previous_overlap = _take_tail(parts, overlap_n)
            blended_overlap = _blend_depth_overlap(previous_overlap, chunk_depth[:overlap_n])
            _replace_tail(parts, blended_overlap.astype(np.float32, copy=False))
            append_depth = chunk_depth[overlap_n:]
        else:
            append_depth = chunk_depth

        if append_depth.shape[0] > 0:
            parts.append(np.ascontiguousarray(append_depth, dtype=np.float32))
            unique_frames += int(append_depth.shape[0])
        if log is not None:
            log(chunk_id, int(append_depth.shape[0]), unique_frames)
        chunks[chunk_id] = None  # type: ignore[call-overload]

    depth = _concatenate_depth_parts(parts)
    if expected_total is not None and depth.shape[0] != expected_total:
        raise RuntimeError(
            f"chunked depth produced T={depth.shape[0]} but expected T={expected_total}"
        )
    return depth


# ------------------------------------------------------- D7: normalisation


def normalize_depth_inplace(
    depth: np.ndarray,
    norm_percentile: float = 0.01,
    max_samples: int = 50_000_000,
) -> Tuple[np.ndarray, float, float]:
    """Normalize in place to [0, 1] and report the (lo, hi) the map used so
    callers can persist them (disp_min/disp_max) for reversing it later.

    `norm_percentile` = p means lo/hi are the p-th and (100-p)-th percentiles
    and values beyond them are clipped; 0 restores the absolute min/max, which
    a single outlier pixel could set (car: p99.99 = 0.898 vs max 1.0) and so
    wasted part of the [0, 1] range and of every quantiser downstream.
    Percentiles are taken on a strided sample of at most `max_samples` values
    so a 1800x576x1024 array is not copied for the partition.
    """
    if norm_percentile < 0 or norm_percentile >= 50:
        raise ValueError("norm_percentile must be in [0, 50)")
    flat = depth.reshape(-1)
    if norm_percentile > 0:
        step = max(1, int(np.ceil(flat.size / float(max_samples))))
        sample = flat[::step]
        sample = sample[np.isfinite(sample)]
        if sample.size == 0:
            depth_min, depth_max = 0.0, 1.0
        else:
            lo, hi = np.percentile(sample, [norm_percentile, 100.0 - norm_percentile])
            depth_min, depth_max = float(lo), float(hi)
    else:
        depth_min = float(np.nanmin(depth))
        depth_max = float(np.nanmax(depth))
    denom = max(depth_max - depth_min, 1e-6)
    depth -= depth_min
    depth /= denom
    np.nan_to_num(depth, copy=False, nan=0.0, posinf=1.0, neginf=0.0)
    np.clip(depth, 0.0, 1.0, out=depth)
    return depth, depth_min, depth_max


def save_depth_npz_with_extras(
    path,
    depth: np.ndarray,
    disp_min: Optional[float],
    disp_max: Optional[float],
    extras: Optional[Dict[str, object]] = None,
) -> None:
    """save_depth_npz (uint16 fixed point) plus small scalar extras such as
    `norm_percentile`. depth_io owns the on-disk format and ignores keys it does
    not know, so the extras are appended to the finished npz -- a zip archive
    -- as further `.npy` members instead of teaching depth_io every caller's
    provenance; np.load lists them like any other key."""
    save_depth_npz(path, depth, disp_min=disp_min, disp_max=disp_max)
    if not extras:
        return
    with zipfile.ZipFile(path, "a", compression=zipfile.ZIP_DEFLATED) as archive:
        for key, value in extras.items():
            buf = io.BytesIO()
            np.lib.format.write_array(buf, np.asarray(value), allow_pickle=False)
            archive.writestr(f"{key}.npy", buf.getvalue())
