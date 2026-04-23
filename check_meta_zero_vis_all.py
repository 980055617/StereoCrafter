#!/usr/bin/env python3
import argparse
import collections
import struct
import zlib
import zipfile
from typing import DefaultDict, Dict, List, Optional, Tuple

try:
    import lz4.frame as lz4f  # type: ignore
except Exception:
    lz4f = None

MAGIC = b"SVB1"
HEADER_V1_STRUCT = struct.Struct("<4sHHHHfIHHfffQ")
HEADER_V2_STRUCT = struct.Struct("<4sHHHHfIHHfffQIQ")


def parse_category_table(data: bytes) -> Dict[int, int]:
    pos = 0
    if len(data) < 2:
        return {}
    (count,) = struct.unpack_from("<H", data, pos)
    pos += 2
    out: Dict[int, int] = {}
    for _ in range(count):
        if pos + 6 > len(data):
            break
        cat_id, kp_count, name_len = struct.unpack_from("<HHH", data, pos)
        pos += 6
        if pos + name_len > len(data):
            break
        pos += name_len
        if pos + 2 > len(data):
            break
        (edges_len,) = struct.unpack_from("<H", data, pos)
        pos += 2 + edges_len * 4
        out[int(cat_id)] = int(kp_count)
    return out


def decode_payload(comp_id: int, data: bytes) -> bytes:
    if comp_id == 0:
        return data
    if comp_id == 1:
        return zlib.decompress(data)
    if comp_id == 2 and lz4f is not None:
        return lz4f.decompress(data)
    raise RuntimeError(f"unsupported compression id={comp_id}")


def inspect_meta(
    meta_path: str,
    frame_from: int,
    frame_to: int,
    track_id_filter: int,
    top_n: int,
) -> None:
    with open(meta_path, "rb") as f:
        prefix = f.read(6)
        if len(prefix) < 6:
            raise RuntimeError("short header")
        magic, version = struct.unpack("<4sH", prefix)
        if magic != MAGIC:
            raise RuntimeError("bad magic")
        f.seek(0)
        if version >= 2:
            hb = f.read(HEADER_V2_STRUCT.size)
            (
                _magic,
                _ver,
                comp_id,
                _width,
                _height,
                _fps,
                num_frames,
                _w_eye,
                _res,
                _fovx,
                _qpos,
                qjoint,
                cat_off,
                cat_size,
                index_off,
            ) = HEADER_V2_STRUCT.unpack(hb)
        else:
            hb = f.read(HEADER_V1_STRUCT.size)
            (
                _magic,
                _ver,
                comp_id,
                _width,
                _height,
                _fps,
                num_frames,
                _w_eye,
                _res,
                _fovx,
                _qpos,
                qjoint,
                index_off,
            ) = HEADER_V1_STRUCT.unpack(hb)
            cat_off = 0
            cat_size = 0

        kp_counts: Dict[int, int] = {}
        if cat_size > 0:
            f.seek(cat_off)
            kp_counts = parse_category_table(f.read(cat_size))

        f.seek(index_off)
        offsets = [struct.unpack("<Q", f.read(8))[0] for _ in range(int(num_frames))]

        hits: List[Tuple[int, int, int, int, int, int, int, int]] = []
        by_track: DefaultDict[int, int] = collections.defaultdict(int)
        by_joint: DefaultDict[int, int] = collections.defaultdict(int)

        lo = max(0, frame_from)
        hi = min(int(num_frames) - 1, frame_to)
        for fi in range(lo, hi + 1):
            off = offsets[fi]
            f.seek(off)
            (clen,) = struct.unpack("<I", f.read(4))
            payload = decode_payload(int(comp_id), f.read(clen))
            pos = 0
            (obj_n,) = struct.unpack_from("<H", payload, pos)
            pos += 2
            for _ in range(obj_n):
                track_id = int(struct.unpack_from("<I", payload, pos)[0])
                pos += 4
                cat_id = int(payload[pos])
                pos += 1
                flags = int(payload[pos])
                pos += 1
                pos += 8 + 4 + 2 + 2 + 8
                if not (flags & 1):
                    continue
                kp_count = int(kp_counts.get(cat_id, 0))
                joints = struct.unpack_from("<" + "h" * (kp_count * 3), payload, pos)
                pos += kp_count * 3 * 2
                vis = struct.unpack_from("<" + "B" * kp_count, payload, pos)
                pos += kp_count
                if track_id_filter >= 0 and track_id != track_id_filter:
                    continue
                for ji in range(kp_count):
                    x = int(joints[ji * 3 + 0])
                    y = int(joints[ji * 3 + 1])
                    z = int(joints[ji * 3 + 2])
                    v = int(vis[ji])
                    if x == 0 and y == 0 and z == 0 and v > 0:
                        hits.append((fi, track_id, cat_id, ji, v, x, y, z))
                        by_track[track_id] += 1
                        by_joint[ji] += 1

        print(f"header: frames={num_frames} comp={comp_id} qjoint={qjoint}")
        print(f"scan_range: {lo}..{hi} track_filter={track_id_filter}")
        print(f"hits_total={len(hits)}")
        print(f"top{top_n}:")
        for fi, tid, cat, ji, v, x, y, z in hits[:top_n]:
            print(
                f"hit frame={fi} trackId={tid} cat={cat} jointIdx={ji} vis={v} "
                f"q=({x},{y},{z}) raw=({x*qjoint:.4f},{y*qjoint:.4f},{z*qjoint:.4f})"
            )
        print("by_track:")
        for tid in sorted(by_track.keys()):
            print(f"  trackId={tid} hits={by_track[tid]}")
        print("by_joint:")
        for ji in sorted(by_joint.keys()):
            print(f"  jointIdx={ji} hits={by_joint[ji]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--frame_from", type=int, default=0)
    ap.add_argument("--frame_to", type=int, default=10**9)
    ap.add_argument("--track_id", type=int, default=-1)
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()

    with zipfile.ZipFile(args.bundle, "r") as zf:
        meta = zf.read("meta.bin")
    tmp = "/tmp/_meta_scan_all.bin"
    with open(tmp, "wb") as f:
        f.write(meta)
    inspect_meta(tmp, args.frame_from, args.frame_to, args.track_id, args.top)


if __name__ == "__main__":
    main()
