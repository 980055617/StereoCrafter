#!/usr/bin/env python3
import argparse
import json
import os
import struct
import sys
import zipfile
import zlib
from typing import Any, Dict, List, Tuple

try:
    import lz4.frame as lz4f  # type: ignore
except Exception:
    lz4f = None


HEADER_V1_STRUCT = struct.Struct("<4sHHHHfIHHfffQ")
HEADER_V2_STRUCT = struct.Struct("<4sHHHHfIHHfffQIQ")
MAGIC = b"SVB1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify SVB bundle integrity.")
    parser.add_argument("--bundle", required=True, help="Path to bundle.svb")
    parser.add_argument("--max_frames", type=int, default=300, help="Max frames to decode for validation.")
    parser.add_argument("--dump_frame", type=int, default=None, help="Frame index to dump objects.")
    parser.add_argument("--out_json", default=None, help="Write summary to JSON.")
    return parser.parse_args()


def fail(stage: str, message: str) -> None:
    raise RuntimeError(f"[{stage}] {message}")


def warn(warnings: List[str], stage: str, message: str) -> None:
    msg = f"[{stage}] {message}"
    warnings.append(msg)
    print(f"Warning {msg}")


def parse_header(data: bytes) -> Dict[str, Any]:
    if len(data) < HEADER_V1_STRUCT.size:
        fail("header", f"meta.bin too small ({len(data)} bytes).")
    (
        magic,
        version,
        compress_id,
        width,
        height,
        fps,
        num_frames,
        w_eye,
        reserved,
        fovx_deg,
        quant_pos_scale,
        quant_joint_scale,
        index_table_offset,
    ) = HEADER_V1_STRUCT.unpack_from(data, 0)
    header = {
        "magic": magic,
        "version": version,
        "compress_id": compress_id,
        "width": width,
        "height": height,
        "fps": fps,
        "num_frames": num_frames,
        "w_eye": w_eye,
        "reserved": reserved,
        "fovx_deg": fovx_deg,
        "quant_pos_scale": quant_pos_scale,
        "quant_joint_scale": quant_joint_scale,
        "index_table_offset": index_table_offset,
        "category_table_offset": 0,
        "category_table_size": 0,
    }
    if version >= 2:
        if len(data) < HEADER_V2_STRUCT.size:
            fail("header", f"meta.bin too small for v2 ({len(data)} bytes).")
        (
            magic,
            version,
            compress_id,
            width,
            height,
            fps,
            num_frames,
            w_eye,
            reserved,
            fovx_deg,
            quant_pos_scale,
            quant_joint_scale,
            category_table_offset,
            category_table_size,
            index_table_offset,
        ) = HEADER_V2_STRUCT.unpack_from(data, 0)
        header.update(
            {
                "magic": magic,
                "version": version,
                "compress_id": compress_id,
                "width": width,
                "height": height,
                "fps": fps,
                "num_frames": num_frames,
                "w_eye": w_eye,
                "reserved": reserved,
                "fovx_deg": fovx_deg,
                "quant_pos_scale": quant_pos_scale,
                "quant_joint_scale": quant_joint_scale,
                "category_table_offset": category_table_offset,
                "category_table_size": category_table_size,
                "index_table_offset": index_table_offset,
            }
        )
    return header


def validate_header(header: Dict[str, Any], meta_size: int) -> None:
    if header["magic"] != MAGIC:
        fail("header", f"magic mismatch: {header['magic']!r}")
    if header["version"] not in (1, 2):
        fail("header", f"unsupported version: {header['version']}")
    compress_id = header["compress_id"]
    if compress_id not in (0, 1, 2):
        fail("header", f"unknown compress_id: {compress_id}")
    width = header["width"]
    if width % 2 != 0:
        fail("header", f"width must be even for SBS: {width}")
    if header["w_eye"] != width // 2:
        fail("header", f"w_eye mismatch: {header['w_eye']} vs {width // 2}")
    if header["fps"] <= 0:
        fail("header", f"fps must be > 0, got {header['fps']}")
    if header["num_frames"] <= 0:
        fail("header", f"num_frames must be > 0, got {header['num_frames']}")
    if header["quant_pos_scale"] <= 0 or header["quant_joint_scale"] <= 0:
        fail("header", "quant scales must be > 0")
    if header["version"] >= 2:
        cat_off = header["category_table_offset"]
        cat_size = header["category_table_size"]
        if cat_off >= meta_size:
            fail(
                "header",
                f"category_table_offset out of range: {cat_off} (size {meta_size})",
            )
        if cat_size < 0 or cat_off + cat_size > meta_size:
            fail(
                "header",
                f"category_table range invalid: offset {cat_off} size {cat_size} (size {meta_size})",
            )
    if header["index_table_offset"] >= meta_size:
        fail(
            "header",
            f"index_table_offset out of range: {header['index_table_offset']} (size {meta_size})",
        )


def default_category_specs(version: int) -> Dict[int, Dict[str, Any]]:
    if version == 1:
        return {
            0: {"name": "human", "kp_count": 17},
            1: {"name": "animal", "kp_count": 20},
            2: {"name": "rigid", "kp_count": 0},
        }
    return {
        0: {"name": "other", "kp_count": 0},
        1: {"name": "person", "kp_count": 17},
        2: {"name": "animal", "kp_count": 20},
    }


def parse_category_table(
    data: bytes, offset: int, size: int, warnings: List[str]
) -> Dict[int, Dict[str, Any]]:
    if size <= 0:
        return {}
    if offset + size > len(data):
        fail("category", "category table extends beyond file size")
    pos = offset
    if pos + 2 > offset + size:
        fail("category", "category table truncated (missing count)")
    num_categories = struct.unpack_from("<H", data, pos)[0]
    pos += 2
    specs: Dict[int, Dict[str, Any]] = {}
    for _ in range(num_categories):
        if pos + 6 > offset + size:
            fail("category", "category table truncated (entry header)")
        cat_id, kp_count, name_len = struct.unpack_from("<HHH", data, pos)
        pos += 6
        if pos + name_len > offset + size:
            fail("category", "category table truncated (name)")
        name = data[pos : pos + name_len].decode("utf-8", errors="replace")
        pos += name_len
        if pos + 2 > offset + size:
            fail("category", "category table truncated (edge count)")
        edge_count = struct.unpack_from("<H", data, pos)[0]
        pos += 2
        edges = []
        for _ in range(edge_count):
            if pos + 4 > offset + size:
                fail("category", "category table truncated (edges)")
            a, b = struct.unpack_from("<HH", data, pos)
            pos += 4
            edges.append((a, b))
        specs[int(cat_id)] = {"name": name, "kp_count": int(kp_count), "skeleton_edges": edges}
    if pos != offset + size:
        warn(warnings, "category", f"category table has {offset + size - pos} trailing bytes")
    return specs


def read_index_table(
    data: bytes, num_frames: int, index_table_offset: int, meta_size: int
) -> List[int]:
    table_size = num_frames * 8
    if index_table_offset + table_size > meta_size:
        fail("index", "index table extends beyond file size")
    offsets = []
    pos = index_table_offset
    for i in range(num_frames):
        off = struct.unpack_from("<Q", data, pos)[0]
        pos += 8
        offsets.append(off)
    for i in range(1, len(offsets)):
        if offsets[i] < offsets[i - 1]:
            fail("index", f"offsets not monotonic at {i}: {offsets[i - 1]} -> {offsets[i]}")
    for i, off in enumerate(offsets):
        if off >= meta_size:
            fail("index", f"offset[{i}] out of range: {off}")
    return offsets


def decompress_payload(compress_id: int, chunk: bytes) -> bytes:
    if compress_id == 0:
        return chunk
    if compress_id == 1:
        return zlib.decompress(chunk)
    if compress_id == 2:
        if lz4f is None:
            fail("frame", "lz4 not available. Install with: pip install lz4")
        return lz4f.decompress(chunk)
    fail("frame", f"unknown compress_id: {compress_id}")
    return b""


def parse_payload(
    payload: bytes,
    warnings: List[str],
    frame_idx: int,
    cat_specs: Dict[int, Dict[str, Any]],
    dump: bool = False,
    dump_limit: int = 5,
) -> Tuple[int, int, Dict[int, int], int, int, int, int, int]:
    view = memoryview(payload)
    pos = 0
    if len(payload) < 2:
        fail("frame", f"frame {frame_idx} payload too small")
    num_objects = struct.unpack_from("<H", view, pos)[0]
    pos += 2
    type_counts: Dict[int, int] = {}
    skeleton_count = 0
    bbox_zero = 0
    anchor_zero = 0
    unknown_type = 0
    vis_total = 0
    vis_invalid = 0

    for obj_idx in range(num_objects):
        base_len = 30
        if pos + base_len > len(payload):
            fail("frame", f"frame {frame_idx} truncated at object {obj_idx}")
        track_id = struct.unpack_from("<I", view, pos)[0]
        pos += 4
        obj_type = view[pos]
        pos += 1
        flags = view[pos]
        pos += 1
        bbox = struct.unpack_from("<HHHH", view, pos)
        pos += 8
        anchor_u, anchor_v = struct.unpack_from("<HH", view, pos)
        pos += 4
        anchor_z_q = struct.unpack_from("<h", view, pos)[0]
        pos += 2
        anchor_scale_q = struct.unpack_from("<H", view, pos)[0]
        pos += 2
        rot_q = struct.unpack_from("<hhhh", view, pos)
        pos += 8

        if bbox[2] == 0 or bbox[3] == 0:
            bbox_zero += 1
        if anchor_u == 0 and anchor_v == 0:
            anchor_zero += 1

        spec = cat_specs.get(obj_type)
        if spec is None:
            unknown_type += 1
        type_counts[obj_type] = type_counts.get(obj_type, 0) + 1

        has_skel = (flags & 1) == 1
        if has_skel:
            skeleton_count += 1
            kp_count = int(spec.get("kp_count", 0)) if spec else 0
            if kp_count <= 0:
                fail(
                    "frame",
                    f"frame {frame_idx} obj {obj_idx} has_skel but kp_count=0 (type {obj_type})",
                )
            skel_len = kp_count * 3 * 2 + kp_count
            if pos + skel_len > len(payload):
                fail("frame", f"frame {frame_idx} skeleton truncated at object {obj_idx}")
            joints_bytes = kp_count * 3 * 2
            conf_view = view[pos + joints_bytes : pos + skel_len]
            vis_total += kp_count
            for v in conf_view:
                if v not in (0, 1, 2):
                    vis_invalid += 1
            pos += skel_len

        if dump and obj_idx < dump_limit:
            type_name = spec.get("name") if spec else f"unknown({obj_type})"
            skel_n = int(spec.get("kp_count", 0)) if spec else 0
            print(
                f"  obj[{obj_idx}] track_id={track_id} type={type_name} "
                f"bbox={bbox} uv=({anchor_u},{anchor_v}) z_q={anchor_z_q} "
                f"has_skel={int(has_skel)} skel_n={skel_n} scale_q={anchor_scale_q} rot_q={rot_q}"
            )

    if pos != len(payload):
        warn(warnings, "frame", f"frame {frame_idx} has {len(payload) - pos} trailing bytes")

    return (
        num_objects,
        skeleton_count,
        type_counts,
        bbox_zero,
        anchor_zero,
        unknown_type,
        vis_total,
        vis_invalid,
    )


def decode_frame(
    data: bytes,
    offsets: List[int],
    frame_idx: int,
    compress_id: int,
    warnings: List[str],
    cat_specs: Dict[int, Dict[str, Any]],
    dump: bool = False,
) -> Tuple[int, int, Dict[int, int], int, int, int, int, int]:
    if frame_idx >= len(offsets):
        fail("frame", f"frame index {frame_idx} out of range")
    off = offsets[frame_idx]
    if off + 4 > len(data):
        fail("frame", f"frame {frame_idx} offset out of range")
    chunk_len = struct.unpack_from("<I", data, off)[0]
    off += 4
    if off + chunk_len > len(data):
        fail("frame", f"frame {frame_idx} chunk exceeds file size")
    chunk = data[off : off + chunk_len]
    payload = decompress_payload(compress_id, chunk)
    return parse_payload(payload, warnings, frame_idx, cat_specs, dump=dump)


def main() -> None:
    args = parse_args()
    warnings: List[str] = []

    if not os.path.exists(args.bundle):
        fail("zip", f"bundle not found: {args.bundle}")

    print(f"[zip] Open bundle: {args.bundle}")
    try:
        with zipfile.ZipFile(args.bundle, "r") as zf:
            bad_file = zf.testzip()
            if bad_file:
                fail("zip", f"CRC error in {bad_file}")
            names = set(zf.namelist())
            if "meta.bin" not in names:
                fail("zip", "meta.bin missing in bundle")
            if "video.mp4" not in names:
                warn(warnings, "zip", "video.mp4 missing in bundle")

            with zf.open("meta.bin") as handle:
                meta_data = handle.read()
    except zipfile.BadZipFile as exc:
        fail("zip", f"invalid zip: {exc}")

    meta_size = len(meta_data)
    print(f"[zip] meta.bin size: {meta_size} bytes")

    header = parse_header(meta_data)
    validate_header(header, meta_size)

    compress_id = header["compress_id"]
    cat_specs = default_category_specs(header["version"])
    if header["version"] >= 2 and header["category_table_size"] > 0:
        parsed_specs = parse_category_table(
            meta_data,
            header["category_table_offset"],
            header["category_table_size"],
            warnings,
        )
        for cat_id, spec in parsed_specs.items():
            cat_specs[cat_id] = spec
    elif header["version"] >= 2:
        warn(warnings, "category", "category table missing (size=0)")
    index_table_offset = header["index_table_offset"]
    num_frames = header["num_frames"]

    offsets = read_index_table(meta_data, num_frames, index_table_offset, meta_size)
    print(f"[index] offsets read: {len(offsets)}")

    if any(off >= index_table_offset for off in offsets):
        warn(warnings, "index", "some frame offsets point inside index table region")

    max_frames = args.max_frames if args.max_frames is not None else num_frames
    if max_frames < 0:
        max_frames = 0
    checked_frames = min(max_frames, num_frames)
    print(f"[frame] decode up to {checked_frames} frames (of {num_frames})")

    total_objects = 0
    total_skeleton = 0
    total_type_counts: Dict[int, int] = {}
    bbox_zero_count = 0
    anchor_zero_count = 0
    unknown_type_count = 0
    vis_total = 0
    vis_invalid = 0
    bad_frames = 0

    for frame_idx in range(checked_frames):
        try:
            dump = args.dump_frame is not None and frame_idx == args.dump_frame
            if dump:
                print(f"[dump] frame {frame_idx}")
            (
                num_objects,
                skeleton_count,
                type_counts,
                bbox_zero,
                anchor_zero,
                unknown_type,
                vis_total_frame,
                vis_invalid_frame,
            ) = decode_frame(meta_data, offsets, frame_idx, compress_id, warnings, cat_specs, dump=dump)
        except Exception as exc:
            bad_frames += 1
            print(f"[frame] failed at {frame_idx}: {exc}")
            continue
        total_objects += num_objects
        total_skeleton += skeleton_count
        bbox_zero_count += bbox_zero
        anchor_zero_count += anchor_zero
        unknown_type_count += unknown_type
        vis_total += vis_total_frame
        vis_invalid += vis_invalid_frame
        for t_id, count in type_counts.items():
            total_type_counts[t_id] = total_type_counts.get(t_id, 0) + count

    if args.dump_frame is not None and args.dump_frame >= checked_frames:
        if args.dump_frame >= num_frames:
            warn(warnings, "dump", f"dump_frame {args.dump_frame} out of range (0..{num_frames - 1})")
        else:
            print(f"[dump] frame {args.dump_frame}")
            try:
                decode_frame(meta_data, offsets, args.dump_frame, compress_id, warnings, cat_specs, dump=True)
            except Exception as exc:
                print(f"[dump] failed: {exc}")

    good_frames = max(0, checked_frames - bad_frames)
    avg_objects = total_objects / good_frames if good_frames else 0.0
    avg_skeleton = total_skeleton / good_frames if good_frames else 0.0
    avg_type: Dict[str, float] = {}
    for cat_id, count in total_type_counts.items():
        name = cat_specs.get(cat_id, {}).get("name", f"cat_{cat_id}")
        avg_type[name] = count / good_frames if good_frames else 0.0

    if bbox_zero_count > 0:
        warn(warnings, "payload", f"bbox w/h == 0 count: {bbox_zero_count}")
    if total_objects > 0:
        anchor_zero_ratio = anchor_zero_count / total_objects
        if anchor_zero_ratio > 0.3:
            warn(
                warnings,
                "payload",
                f"anchor_uv (0,0) ratio high: {anchor_zero_ratio:.2%} ({anchor_zero_count}/{total_objects})",
            )
    if vis_total > 0:
        vis_ratio = vis_invalid / vis_total
        if vis_ratio > 0.1:
            warn(
                warnings,
                "payload",
                f"visibility values outside 0/1/2 ratio high: {vis_ratio:.2%} ({vis_invalid}/{vis_total})",
            )
    if unknown_type_count > 0:
        warn(warnings, "payload", f"unknown type count: {unknown_type_count}")

    summary = {
        "checked_frames": checked_frames,
        "total_frames": num_frames,
        "bad_frames": bad_frames,
        "avg_objects_per_frame": avg_objects,
        "avg_skeleton_objects_per_frame": avg_skeleton,
        "avg_type_per_frame": avg_type,
        "warning_count": len(warnings),
        "bbox_zero_count": bbox_zero_count,
        "anchor_zero_count": anchor_zero_count,
        "unknown_type_count": unknown_type_count,
        "visibility_total": vis_total,
        "visibility_invalid": vis_invalid,
    }

    print("[summary]")
    print(f"  checked_frames: {checked_frames} / {num_frames}")
    print(f"  bad_frames: {bad_frames}")
    print(f"  avg_objects_per_frame: {avg_objects:.2f}")
    print(f"  avg_skeleton_objects_per_frame: {avg_skeleton:.2f}")
    print(f"  avg_type_per_frame: {avg_type}")
    print(f"  warnings: {len(warnings)}")

    if args.out_json:
        out_dir = os.path.dirname(os.path.abspath(args.out_json)) or "."
        os.makedirs(out_dir, exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        print(f"[summary] wrote {args.out_json}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
