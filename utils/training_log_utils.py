# =============================================
# File: /workspace/stereocraft/utils/training_log_utils.py
# ---------------------------------------------
# 目的: 学習ログ/CSV整形とスナップショット
# =============================================

from __future__ import annotations

import csv
import glob
import json
import logging
import os
from datetime import datetime
from typing import Any, Sequence

from utils.logging_utils import ensure_logging_configured


def resolve_run_save_dir(save_dir: str) -> str:
    """Return a per-run save directory under the base save_dir."""
    if not save_dir:
        return save_dir
    normalized = os.path.normpath(save_dir)
    base_name = os.path.basename(normalized)
    if base_name.startswith("MambaCrafter_"):
        return save_dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(save_dir, f"MambaCrafter_{timestamp}")


def write_run_config_snapshot(
    save_dir: str,
    config_values: dict[str, Any],
    resume_candidate: str | None,
    *,
    logger: logging.Logger | None = None,
) -> None:
    """Persist the resolved training config for this run into save_dir."""
    if not save_dir:
        return
    ensure_logging_configured()
    logger = logger or logging.getLogger(__name__)
    os.makedirs(save_dir, exist_ok=True)
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(save_dir, f"train_config_{run_tag}.json")
    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "resume_candidate": resume_candidate,
        "config": config_values,
    }
    try:
        with open(path, "w", encoding="utf-8") as fp:
            json.dump(payload, fp, indent=2, sort_keys=True, ensure_ascii=True, default=str)
        logger.info("Saved training config snapshot to %s", path)
    except Exception as err:
        logger.warning("Failed to save training config snapshot: %s", err)


def _find_existing_log(save_dir: str, prefix: str, logger: logging.Logger) -> str | None:
    if not save_dir:
        return None
    pattern = os.path.join(save_dir, f"{prefix}_*.csv")
    matches = [path for path in glob.glob(pattern) if os.path.isfile(path)]
    if not matches:
        return None
    if len(matches) > 1:
        latest = max(matches, key=os.path.getmtime)
        logger.warning(
            "Multiple %s logs found in %s; using most recent: %s",
            prefix,
            save_dir,
            latest,
        )
        return latest
    return matches[0]


def select_log_path(
    save_dir: str,
    prefix: str,
    run_tag: str,
    *,
    reuse_existing: bool,
    logger: logging.Logger | None = None,
) -> tuple[str, bool]:
    """Return (csv_path, reused_existing)."""
    logger = logger or logging.getLogger(__name__)
    if reuse_existing:
        existing = _find_existing_log(save_dir, prefix, logger)
        if existing:
            return existing, True
    return os.path.join(save_dir, f"{prefix}_{run_tag}.csv"), False


def init_csv_log(csv_path: str, header: Sequence[str]) -> None:
    if not os.path.exists(csv_path):
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(list(header))


def trim_csv_log_by_epoch(
    csv_path: str,
    *,
    epoch_index: int,
    min_epoch: int,
    header: Sequence[str],
) -> int:
    if not os.path.exists(csv_path):
        return 0
    tmp_path = f"{csv_path}.tmp"
    removed = 0
    header_row = list(header)
    wrote_header = False
    saw_row = False
    with open(csv_path, "r", encoding="utf-8", newline="") as src, open(
        tmp_path,
        "w",
        encoding="utf-8",
        newline="",
    ) as dst:
        reader = csv.reader(src)
        writer = csv.writer(dst)
        for row in reader:
            saw_row = True
            if not wrote_header:
                if row == header_row:
                    writer.writerow(row)
                    wrote_header = True
                    continue
                writer.writerow(header_row)
                wrote_header = True
            if not row:
                continue
            try:
                epoch_val = int(row[epoch_index])
            except Exception:
                continue
            if epoch_val < min_epoch:
                writer.writerow(row)
            else:
                removed += 1
        if not saw_row:
            writer.writerow(header_row)
    os.replace(tmp_path, csv_path)
    return removed


def trim_train_log(csv_path: str, min_epoch: int, header: Sequence[str]) -> int:
    return trim_csv_log_by_epoch(
        csv_path,
        epoch_index=1,
        min_epoch=min_epoch,
        header=header,
    )


def trim_val_log(csv_path: str, min_epoch: int, header: Sequence[str]) -> int:
    return trim_csv_log_by_epoch(
        csv_path,
        epoch_index=1,
        min_epoch=min_epoch,
        header=header,
    )
