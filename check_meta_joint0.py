#!/usr/bin/env python3
import argparse
import struct
import zlib
import zipfile
from typing import Dict, List, Tuple

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
    raise RuntimeError(f"unsupported compression id={comp_id} (need zlib/none)")


def inspect(meta_path: str, frame_from: int, frame_to: int, track_id_filter: int) -> None:
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
                _version,
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
                _version,
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

        print(f"header: frames={num_frames} comp={comp_id} qjoint={qjoint}")
        found = 0
        for fi in range(max(0, frame_from), min(int(num_frames), frame_to + 1)):
            off = offsets[fi]
            f.seek(off)
            (clen,) = struct.unpack("<I", f.read(4))
            payload = decode_payload(int(comp_id), f.read(clen))
            pos = 0
            (obj_n,) = struct.unpack_from("<H", payload, pos)
            pos += 2
            for _ in range(obj_n):
                track_id = struct.unpack_from("<I", payload, pos)[0]
                pos += 4
                cat_id = payload[pos]
                pos += 1
                flags = payload[pos]
                pos += 1
                pos += 8 + 4 + 2 + 2 + 8
                if flags & 1:
                    kp_count = kp_counts.get(int(cat_id), 0)
                    joints = struct.unpack_from("<" + "h" * (kp_count * 3), payload, pos)
                    pos += kp_count * 3 * 2
                    vis = struct.unpack_from("<" + "B" * kp_count, payload, pos)
                    pos += kp_count
                    if kp_count <= 0:
                        continue
                    j0 = joints[0:3]
                    v0 = int(vis[0])
                    if (track_id_filter < 0 or int(track_id) == int(track_id_filter)) and j0 == (0, 0, 0) and v0 > 0:
                        found += 1
                        print(
                            f"HIT frame={fi} trackId={track_id} cat={cat_id} "
                            f"j0_q={j0} j0_f=({j0[0]*qjoint:.4f},{j0[1]*qjoint:.4f},{j0[2]*qjoint:.4f}) vis0={v0}"
                        )
                else:
                    pass
        print(f"hits={found}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--frame_from", type=int, default=0)
    ap.add_argument("--frame_to", type=int, default=300)
    ap.add_argument("--track_id", type=int, default=-1)
    args = ap.parse_args()

    with zipfile.ZipFile(args.bundle, "r") as zf:
        meta = zf.read("meta.bin")
    tmp = "/tmp/_meta_check.bin"
    with open(tmp, "wb") as f:
        f.write(meta)
    inspect(tmp, args.frame_from, args.frame_to, args.track_id)


if __name__ == "__main__":
    main()
