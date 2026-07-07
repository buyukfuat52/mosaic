"""Structured report and manifest writers."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from yolo_mosaic.models import ProcessingStats


def write_json_report(
    path: Path,
    stats: ProcessingStats,
    extra: dict[str, object] | None = None,
) -> None:
    """Write processing statistics as JSON."""

    payload: dict[str, object] = {"stats": stats.as_dict()}
    if extra:
        payload.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    """Write a simple CSV manifest."""

    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
