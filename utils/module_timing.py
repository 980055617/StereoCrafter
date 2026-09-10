from __future__ import annotations

import json
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any

import torch


def _split_patterns(raw: str | None, default: list[str]) -> list[str]:
    if raw is None:
        return default
    patterns = [part.strip() for part in raw.split(",") if part.strip()]
    return patterns or default


def install_cuda_module_timer(
    root: torch.nn.Module,
    *,
    include: str | None = None,
    default_include: list[str] | None = None,
) -> tuple[dict[str, Any], list[Any]]:
    """Install CUDA-event forward timers on matching modules.

    The returned state should be passed to `write_cuda_module_timing` after the
    profiled run. The hooks are intentionally generic so the same profiler can
    compare reference attention and Mamba replacement modules.
    """

    patterns = _split_patterns(include, default_include or ["*.attn1"])
    state: dict[str, Any] = {"patterns": patterns, "modules": {}}
    handles: list[Any] = []

    def matches(name: str) -> bool:
        return any(fnmatchcase(name, pattern) or name == pattern for pattern in patterns)

    for name, module in root.named_modules():
        if not name or not matches(name):
            continue
        record = {
            "class": module.__class__.__name__,
            "events": [],
            "inputShapes": [],
        }
        state["modules"][name] = record

        def pre_hook(_module, inputs, *, _record=record):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _record["_active"] = (start, end)
            if len(_record["inputShapes"]) < 3:
                shapes = []
                for value in inputs:
                    if isinstance(value, torch.Tensor):
                        shapes.append(list(value.shape))
                    else:
                        shapes.append(None)
                _record["inputShapes"].append(shapes)

        def post_hook(_module, _inputs, _output, *, _record=record):
            active = _record.pop("_active", None)
            if active is None:
                return
            start, end = active
            end.record()
            _record["events"].append((start, end))

        handles.append(module.register_forward_pre_hook(pre_hook))
        handles.append(module.register_forward_hook(post_hook))

    return state, handles


def write_cuda_module_timing(
    state: dict[str, Any],
    output_path: str | Path,
    *,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    torch.cuda.synchronize()
    modules = []
    for name, record in state["modules"].items():
        timings = [float(start.elapsed_time(end)) for start, end in record["events"]]
        total_ms = sum(timings)
        calls = len(timings)
        modules.append(
            {
                "name": name,
                "class": record["class"],
                "calls": calls,
                "totalMs": total_ms,
                "avgMs": total_ms / calls if calls else 0.0,
                "maxMs": max(timings) if timings else 0.0,
                "firstMs": timings[0] if timings else 0.0,
                "lastMs": timings[-1] if timings else 0.0,
                "timingsMs": timings,
                "inputShapes": record["inputShapes"],
            }
        )
    modules.sort(key=lambda row: row["totalMs"], reverse=True)
    payload = {
        "schema": "master_project.stereocrafter.cuda_module_timing.v1",
        "patterns": state["patterns"],
        "metadata": metadata or {},
        "totalProfiledMs": sum(row["totalMs"] for row in modules),
        "modules": modules,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload
