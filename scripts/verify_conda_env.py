#!/usr/bin/env python
import pathlib
import sys

import torch


def main() -> int:
    print(f"Python: {sys.version.split()[0]}")
    print(f"Torch: {torch.__version__} | CUDA build: {torch.version.cuda} | CUDA available: {torch.cuda.is_available()}")

    try:
        import triton

        print(f"Triton: {triton.__version__}")
    except Exception as exc:
        print(f"Triton import failed: {exc}")
        return 1

    try:
        import mamba_ssm
        import selective_scan_cuda

        print(f"mamba_ssm: {getattr(mamba_ssm, '__version__', 'unknown')}")
        print("selective_scan_cuda: OK")
    except Exception as exc:
        print(f"mamba_ssm/selective_scan_cuda import failed: {exc}")
        return 1

    torch_lib = pathlib.Path(torch.__file__).parent / "lib"
    print(f"Torch lib dir: {torch_lib}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
