import argparse
import pprint
import sys

import torch


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect a .pt checkpoint file")
    parser.add_argument("path", help="Path to the .pt file")
    parser.add_argument(
        "--keys",
        action="store_true",
        help="Only list top-level keys",
    )
    args = parser.parse_args()

    try:
        ckpt = torch.load(args.path, map_location="cpu")
    except Exception as err:
        print(f"Failed to load {args.path}: {err}", file=sys.stderr)
        return 1

    print(f"# Loaded: {args.path}")
    print(f"# Type: {type(ckpt)}")

    if isinstance(ckpt, dict):
        if args.keys:
            print("# Top-level keys:")
            for k in ckpt.keys():
                v = ckpt[k]
                vtype = type(v)
                if isinstance(v, dict):
                    print(f"  - {k}: dict (len={len(v)})")
                elif isinstance(v, list):
                    print(f"  - {k}: list (len={len(v)})")
                else:
                    print(f"  - {k}: {vtype}")
        else:
            pprint.pprint(ckpt)
    else:
        pprint.pprint(ckpt)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
