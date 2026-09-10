"""Write an inference-ready checkpoint with the EMA weights applied.

`inpainting_train.py` stores an EMA shadow of the *trainable* parameters under
`model_ema` (added 2026-09-01). This script merges that shadow over the raw
`model` state dict and writes a new checkpoint.

Deliberately does NOT strip `.mamba_gate` buffers, unlike
`_strip_gated_reference_state` / `*_mamba_only.pt`. That stripping is the cause
of a silent failure mode: without the gate buffers the gate falls back to
`MAMBA_SELF_ATTN_INITIAL_GATE`, which defaults to 0.0, so the Mamba path is
disabled and the model silently runs pure reference attention. Keeping the
gates means the exported file can be handed straight to
`inpainting_inference_hybrid_exclude_up3_attn1.py` with no
`--mamba_gate_override` needed.

Usage:
    python scripts/export_ema_checkpoint.py <train_state_epochNNNNNN.pt> [-o OUT]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", help="train_state_*.pt containing 'model' and 'model_ema'")
    ap.add_argument("-o", "--output", default=None, help="output path (default: <input>_ema.pt)")
    ap.add_argument("--decay", default=None,
                    help="when the checkpoint tracked several decays, pick one "
                         "(e.g. 0.9999); default = the primary shadow")
    args = ap.parse_args()

    src = Path(args.checkpoint)
    ckpt = torch.load(str(src), map_location="cpu")
    if not isinstance(ckpt, dict) or "model" not in ckpt:
        raise SystemExit(f"{src}: not a training checkpoint (no 'model' key)")

    model = ckpt["model"]
    ema = ckpt.get("model_ema")
    by_decay = ckpt.get("model_ema_by_decay") or {}
    if args.decay is not None:
        if args.decay not in by_decay:
            raise SystemExit(
                f"{src}: decay {args.decay} not tracked. available: {sorted(by_decay)}"
            )
        ema = by_decay[args.decay]
    if not ema:
        raise SystemExit(
            f"{src}: no 'model_ema' in checkpoint. Was it trained with --use_ema=True? "
            "(EMA was added 2026-09-01; older checkpoints have none.)"
        )

    merged = dict(model)
    applied = skipped = 0
    for name, shadow in ema.items():
        cur = merged.get(name)
        if cur is None or tuple(cur.shape) != tuple(shadow.shape):
            skipped += 1
            continue
        merged[name] = shadow.to(dtype=cur.dtype)
        applied += 1

    out = Path(args.output) if args.output else src.with_name(src.stem + "_ema.pt")
    torch.save(
        {
            "model": merged,
            "ema_applied": True,
            "ema_num_updates": int(ckpt.get("ema_num_updates", 0) or 0),
            "stage_idx": ckpt.get("stage_idx"),
            "stage_name": ckpt.get("stage_name"),
            "epoch": ckpt.get("epoch"),
        },
        str(out),
    )
    gates = sum(1 for k in merged if k.endswith(".mamba_gate"))
    print(
        f"wrote {out}\n"
        f"  EMA params applied : {applied}  (skipped {skipped})\n"
        f"  ema_num_updates    : {ckpt.get('ema_num_updates')}\n"
        f"  mamba_gate buffers kept : {gates} (so no --mamba_gate_override needed)"
    )


if __name__ == "__main__":
    main()
