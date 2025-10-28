#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Visualize and compare UNet module structures and parameter counts.

This script loads:
  - Base (original) UNet from --base_dir (expects subfolder "unet")
  - Adapted (modified) UNet from --adapted_dir (e.g., checkpoints/.../unet_diffusers)

It then builds a hierarchical module tree for each, aggregates parameter counts
per module (self and total including children), and outputs:
  - Text trees for base, adapted, and a merged diff
  - CSV with per-module parameter diffs

Optional pretty console output uses 'rich' (if installed). The script works
without it by falling back to ASCII trees.

Usage example:
  python scripts/visualize_unet_compare.py \
    --base_dir ./weights/StereoCrafter \
    --adapted_dir ./checkpoints/mamba-unet-overfit/unet_diffusers \
    --output_dir ./reports/unet_compare \
    --max_depth 3

Notes:
  - No forward pass is performed; visualization is static based on module tree
    and parameter tensors, making it memory-safe.
  - If your adapted UNet directory lacks config.json, pass --base_dir_for_config
    so we can reconstruct the UNet before loading the adapted state dict.
  - diff_tree.txt is pruned to include only changed/added/removed nodes and
    their necessary ancestors (ancestors are marked as [context]).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import torch
from diffusers import UNetSpatioTemporalConditionModel

from blocks.mamba_diffusers_adapter import (
    replace_unet_spatiotemporal_transformer_with_mamba,
)


def _count_params(module: torch.nn.Module) -> Tuple[int, int]:
    """Return (self_params, self_trainable_params) not including children."""
    self_params = 0
    self_trainable = 0
    for p in module.parameters(recurse=False):
        n = p.numel()
        self_params += n
        if p.requires_grad:
            self_trainable += n
    return self_params, self_trainable


@dataclass
class ModuleNode:
    name: str
    type: str
    path: str  # dotted path from root; root has ""
    self_params: int = 0
    self_trainable: int = 0
    total_params: int = 0
    total_trainable: int = 0
    children: Dict[str, "ModuleNode"] = field(default_factory=dict)

    def ensure_child(self, child_name: str, child_type: str, child_path: str) -> "ModuleNode":
        if child_name not in self.children:
            self.children[child_name] = ModuleNode(
                name=child_name, type=child_type, path=child_path
            )
        return self.children[child_name]


def build_module_tree(model: torch.nn.Module) -> Tuple[ModuleNode, Dict[str, ModuleNode]]:
    """Collect a hierarchical module tree and a map path->node.

    The root node path is ''. For each named module 'a.b.c', we create nodes
    for 'a', 'a.b', and 'a.b.c'. We collect self-only parameter counts for each
    node, then compute totals bottom-up.
    """
    # Create root
    root = ModuleNode(name="<root>", type=type(model).__name__, path="")
    path2node: Dict[str, ModuleNode] = {"": root}

    # First pass: ensure nodes exist and attach self-only param counts
    for mod_path, module in model.named_modules():
        # named_modules includes root as '' path as well
        path_elems = mod_path.split(".") if mod_path else []
        cur = root
        cur_path = ""
        for i, elem in enumerate(path_elems):
            cur_path = elem if cur_path == "" else f"{cur_path}.{elem}"
            if cur_path not in path2node:
                # type for intermediate node may be unknown until we reach it in named_modules iteration
                # we can read it from model.get_submodule(cur_path)
                sub = model.get_submodule(cur_path)
                path2node[cur_path] = cur.ensure_child(
                    elem, type(sub).__name__, cur_path
                )
                cur = path2node[cur_path]
            else:
                cur = path2node[cur_path]

        # Now cur is the node for this module path (or root)
        if mod_path not in path2node:
            # Should not happen, but be robust
            path2node[mod_path] = cur
        # Update type and self counts
        node = path2node[mod_path]
        node.type = type(module).__name__
        sp, st = _count_params(module)
        node.self_params = sp
        node.self_trainable = st

    # Second pass: compute total params bottom-up
    # Build a list of paths sorted by depth descending
    by_depth = sorted(path2node.keys(), key=lambda p: (0 if p == "" else p.count(".") + 1), reverse=True)
    for p in by_depth:
        node = path2node[p]
        # sum children
        total = node.self_params
        total_tr = node.self_trainable
        for child in node.children.values():
            total += child.total_params
            total_tr += child.total_trainable
        node.total_params = total
        node.total_trainable = total_tr

    return root, path2node


def load_unet_base(base_dir: str) -> UNetSpatioTemporalConditionModel:
    return UNetSpatioTemporalConditionModel.from_pretrained(
        base_dir, subfolder="unet", low_cpu_mem_usage=True, torch_dtype=torch.float32
    )


def load_unet_adapted(adapted_dir: str, base_dir_for_config: Optional[str] = None) -> UNetSpatioTemporalConditionModel:
    # 1) Load config
    cfg_path = os.path.join(adapted_dir, "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path, "r") as f:
            cfg = json.load(f)
        unet = UNetSpatioTemporalConditionModel.from_config(cfg)
    else:
        if base_dir_for_config is None:
            raise FileNotFoundError(
                "config.json not found in adapted_dir; pass --base_dir_for_config"
            )
        unet = UNetSpatioTemporalConditionModel.from_pretrained(
            base_dir_for_config, subfolder="unet", low_cpu_mem_usage=True, torch_dtype=torch.float32
        )

    # 2) Adapter settings (optional)
    adapter_cfg = {
        "mamba_d_state": 128,
        "mamba_expand": 1,
        "temporal_chunk": 32,
        "spatial_chunk": 2048,
        "mem_eff": True,
        "keep_spatial_mixer": True,
    }
    ac_path = os.path.join(adapted_dir, "mamba_adapter.json")
    if os.path.exists(ac_path):
        try:
            with open(ac_path, "r") as f:
                ac = json.load(f)
            adapter_cfg.update(
                {
                    "mamba_d_state": int(ac.get("mamba_d_state", adapter_cfg["mamba_d_state"])),
                    "mamba_expand": int(ac.get("mamba_expand", adapter_cfg["mamba_expand"])),
                    "temporal_chunk": int(ac.get("temporal_chunk", adapter_cfg["temporal_chunk"])),
                    "spatial_chunk": int(ac.get("spatial_chunk", adapter_cfg["spatial_chunk"])),
                    "mem_eff": bool(ac.get("mem_eff", adapter_cfg["mem_eff"])),
                }
            )
        except Exception:
            pass

    # 3) Replace transformer with Mamba adapter
    replace_unet_spatiotemporal_transformer_with_mamba(
        unet,
        use_mem_eff_path=adapter_cfg["mem_eff"] and torch.cuda.is_available(),
        temporal_chunk_size=adapter_cfg["temporal_chunk"],
        spatial_chunk_size=adapter_cfg["spatial_chunk"],
        d_state=adapter_cfg["mamba_d_state"],
        expand=adapter_cfg["mamba_expand"],
        keep_spatial_mixer=adapter_cfg["keep_spatial_mixer"],
    )

    # 4) Load weights
    cand = [
        os.path.join(adapted_dir, "diffusion_pytorch_model.safetensors"),
        os.path.join(adapted_dir, "model.safetensors"),
        os.path.join(adapted_dir, "diffusion_pytorch_model.bin"),
        os.path.join(adapted_dir, "pytorch_model.bin"),
    ]
    sd_path_st = next((p for p in cand if p.endswith(".safetensors") and os.path.exists(p)), None)
    sd_path_pt = next((p for p in cand if p.endswith(".bin") and os.path.exists(p)), None)
    if sd_path_st is not None:
        from safetensors.torch import load_file as load_sft
        sd = load_sft(sd_path_st, device="cpu")
    elif sd_path_pt is not None:
        sd = torch.load(sd_path_pt, map_location="cpu")
    else:
        raise FileNotFoundError(f"No state dict found in {adapted_dir}")
    unet.load_state_dict(sd, strict=False)
    return unet


@dataclass
class DiffNode:
    name: str
    path: str
    base_type: Optional[str]
    adapted_type: Optional[str]
    base_total: int
    adapted_total: int
    delta: int
    status: str  # 'same' | 'changed' | 'added' | 'removed' | 'context'
    children: List["DiffNode"] = field(default_factory=list)


def build_diff_tree(base_root: ModuleNode, base_map: Dict[str, ModuleNode],
                    ad_root: ModuleNode, ad_map: Dict[str, ModuleNode]) -> DiffNode:
    def children_names(node: Optional[ModuleNode]) -> List[str]:
        if node is None:
            return []
        return sorted(node.children.keys())

    def make(path: str) -> DiffNode:
        b = base_map.get(path)
        a = ad_map.get(path)
        name = path.split(".")[-1] if path else "<root>"
        base_type = b.type if b else None
        ad_type = a.type if a else None
        btot = b.total_params if b else 0
        atot = a.total_params if a else 0
        delta = atot - btot
        if b and a:
            if base_type == ad_type and delta == 0:
                status = "same"
            else:
                status = "changed"
        elif a and not b:
            status = "added"
        else:
            status = "removed"
        node = DiffNode(
            name=name, path=path, base_type=base_type, adapted_type=ad_type,
            base_total=btot, adapted_total=atot, delta=delta, status=status
        )
        # collect child names
        bnode = base_map.get(path)
        anode = ad_map.get(path)
        child_set = set(children_names(bnode)) | set(children_names(anode))
        node.children = [make(path + ("." if path else "") + c) for c in sorted(child_set)]
        return node

    return make("")


def render_ascii_tree(node: DiffNode, max_depth: int = 3, show_only_changed: bool = False) -> str:
    lines: List[str] = []

    def fmt_counts(b: int, a: int) -> str:
        sign = "+" if (a - b) >= 0 else "-"
        delta = abs(a - b)
        return f"{b:,} -> {a:,} ({sign}{delta:,})"

    def fmt_types(bt: Optional[str], at: Optional[str]) -> str:
        if bt == at:
            return bt or "-"
        return f"{bt or '-'} -> {at or '-'}"

    def rec(n: DiffNode, depth: int, prefix: str, is_last: bool):
        if show_only_changed and n.status == "same":
            # Skip only if truly unchanged and we are not a context node.
            return
        branch = "└─" if is_last else "├─"
        pad = prefix + branch
        # Compose label
        label = f"{n.name}  [params: {fmt_counts(n.base_total, n.adapted_total)}]  [type: {fmt_types(n.base_type, n.adapted_type)}]  [{n.status}]"
        lines.append(pad + label)
        if depth >= max_depth:
            return
        child_prefix = prefix + ("   " if is_last else "│  ")
        for i, c in enumerate(n.children):
            rec(c, depth + 1, child_prefix, i == len(n.children) - 1)

    # Root line
    header = f"{node.name}  [params: {node.base_total:,} -> {node.adapted_total:,} ({('+' if node.delta>=0 else '-')}{abs(node.delta):,})]  [type: {node.base_type or '-'} -> {node.adapted_type or '-'}]  [{node.status}]"
    lines.append(header)
    for i, ch in enumerate(node.children):
        rec(ch, 1, "", i == len(node.children) - 1)
    return "\n".join(lines)


def prune_diff_tree(node: DiffNode, include_context: bool = True) -> Optional[DiffNode]:
    """Return a pruned copy that keeps only changed/added/removed nodes and
    their necessary ancestors. Ancestors kept are marked as 'context' when
    include_context is True.
    """
    changed_kinds = {"changed", "added", "removed"}

    pruned_children: List[DiffNode] = []
    for c in node.children:
        pc = prune_diff_tree(c, include_context=include_context)
        if pc is not None:
            pruned_children.append(pc)

    is_changed = node.status in changed_kinds
    if not is_changed and not pruned_children:
        return None

    # Keep this node (changed or needed ancestor)
    kept = DiffNode(
        name=node.name,
        path=node.path,
        base_type=node.base_type,
        adapted_type=node.adapted_type,
        base_total=node.base_total,
        adapted_total=node.adapted_total,
        delta=node.delta,
        status=node.status,
        children=pruned_children,
    )
    if include_context and not is_changed:
        kept.status = "context"
    return kept


def write_csv_diff(node: DiffNode, csv_path: str) -> None:
    rows: List[List[str]] = []

    def rec(n: DiffNode):
        rows.append([
            n.path or "<root>",
            n.base_type or "-",
            n.adapted_type or "-",
            str(n.base_total),
            str(n.adapted_total),
            str(n.delta),
            n.status,
        ])
        for c in n.children:
            rec(c)

    rec(node)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "base_type", "adapted_type", "base_params", "adapted_params", "delta", "status"])
        w.writerows(rows)


def try_rich_print(diff_root: DiffNode, max_depth: int, show_only_changed: bool) -> bool:
    try:
        from rich.console import Console
        from rich.tree import Tree
        from rich.text import Text

        def fmt_counts(b: int, a: int) -> str:
            sign = "+" if (a - b) >= 0 else "-"
            delta = abs(a - b)
            return f"{b:,} -> {a:,} ({sign}{delta:,})"

        def fmt_types(bt: Optional[str], at: Optional[str]) -> str:
            if bt == at:
                return bt or "-"
            return f"{bt or '-'} -> {at or '-'}"

        def add(tree: Tree, n: DiffNode, depth: int):
            if show_only_changed and n.status == "same":
                return
            label = Text(n.name)
            label.append(f"  [params: {fmt_counts(n.base_total, n.adapted_total)}]", style="cyan")
            label.append(f"  [type: {fmt_types(n.base_type, n.adapted_type)}]", style="magenta")
            st_style = {
                "same": "dim",
                "context": "dim",
                "changed": "yellow",
                "added": "green",
                "removed": "red",
            }.get(n.status, "white")
            label.append(f"  [{n.status}]", style=st_style)
            node = tree.add(label)
            if depth >= max_depth:
                return
            for c in n.children:
                add(node, c, depth + 1)

        console = Console()
        root_label = Text("<root>")
        root_label.append(
            f"  [params: {diff_root.base_total:,} -> {diff_root.adapted_total:,} ({('+' if diff_root.delta>=0 else '-')}{abs(diff_root.delta):,})]",
            style="cyan",
        )
        root_label.append(
            f"  [type: {diff_root.base_type or '-'} -> {diff_root.adapted_type or '-'}]",
            style="magenta",
        )
        root_label.append(f"  [{diff_root.status}]", style="yellow")
        tree = Tree(root_label)
        for c in diff_root.children:
            add(tree, c, 1)
        console.print(tree)
        return True
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser(description="Visualize and diff UNet module trees")
    ap.add_argument("--base_dir", type=str, required=True, help="Path to base weights dir (contains subfolder 'unet')")
    ap.add_argument("--adapted_dir", type=str, required=True, help="Path to adapted UNet dir (e.g., .../unet_diffusers)")
    ap.add_argument("--base_dir_for_config", type=str, default=None, help="Optional fallback for config if adapted has none")
    ap.add_argument("--output_dir", type=str, default="./reports/unet_compare", help="Directory for output reports")
    ap.add_argument("--max_depth", type=int, default=3, help="Max depth to render in tree outputs")
    ap.add_argument("--show_only_changed", action="store_true", help="Show only changed/added/removed nodes in trees")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load models
    print("[1/4] Loading base UNet...")
    base_unet = load_unet_base(args.base_dir)
    print("[2/4] Loading adapted UNet...")
    adapted_unet = load_unet_adapted(args.adapted_dir, args.base_dir_for_config or args.base_dir)

    # Build trees
    print("[3/4] Building module trees...")
    base_root, base_map = build_module_tree(base_unet)
    ad_root, ad_map = build_module_tree(adapted_unet)

    # Diff
    print("[4/4] Creating diff...")
    diff_root = build_diff_tree(base_root, base_map, ad_root, ad_map)

    # Write outputs: diff_tree is pruned to only changed parts (+ minimal context)
    pruned = prune_diff_tree(diff_root, include_context=True)
    if pruned is None:
        ascii_diff = "<no changes>"
    else:
        ascii_diff = render_ascii_tree(pruned, max_depth=args.max_depth, show_only_changed=False)
    diff_txt = os.path.join(args.output_dir, "diff_tree.txt")
    with open(diff_txt, "w", encoding="utf-8") as f:
        f.write(ascii_diff + "\n")

    # Base and adapted trees (without diff; render just counts)
    # Reuse DiffNode to render per-model trees by comparing to zero baseline
    def wrap_single(root: ModuleNode, label: str) -> DiffNode:
        return DiffNode(
            name=label,
            path="",
            base_type=root.type,
            adapted_type=root.type,
            base_total=root.total_params,
            adapted_total=root.total_params,
            delta=0,
            status="same",
            children=[
                DiffNode(
                    name=child.name,
                    path=child.path,
                    base_type=child.type,
                    adapted_type=child.type,
                    base_total=child.total_params,
                    adapted_total=child.total_params,
                    delta=0,
                    status="same",
                    children=[],
                )
                for child in root.children.values()
            ],
        )

    with open(os.path.join(args.output_dir, "base_tree.txt"), "w", encoding="utf-8") as f:
        f.write(render_ascii_tree(wrap_single(base_root, "<base>"), max_depth=args.max_depth) + "\n")
    with open(os.path.join(args.output_dir, "adapted_tree.txt"), "w", encoding="utf-8") as f:
        f.write(render_ascii_tree(wrap_single(ad_root, "<adapted>"), max_depth=args.max_depth) + "\n")

    # CSV diff
    write_csv_diff(diff_root, os.path.join(args.output_dir, "modules_diff.csv"))

    # Console preview (rich if available)
    print("\n--- Diff Tree (console preview) ---")
    used_rich = False
    if pruned is not None:
        used_rich = try_rich_print(pruned, max_depth=args.max_depth, show_only_changed=False)
    if not used_rich:
        print(ascii_diff)

    print("\nOutputs written to:")
    print(f" - {diff_txt}")
    print(f" - {os.path.join(args.output_dir, 'base_tree.txt')}")
    print(f" - {os.path.join(args.output_dir, 'adapted_tree.txt')}")
    print(f" - {os.path.join(args.output_dir, 'modules_diff.csv')}")


if __name__ == "__main__":
    main()
