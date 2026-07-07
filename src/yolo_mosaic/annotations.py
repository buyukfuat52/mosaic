"""YOLO annotation parsing and serialization."""

from __future__ import annotations

from pathlib import Path

from yolo_mosaic.models import ProcessingStats, YoloBox


class AnnotationError(ValueError):
    """Raised when an annotation file cannot be parsed under strict policy."""


def parse_yolo_line(line: str) -> YoloBox:
    """Parse one YOLO annotation row into a :class:`YoloBox`.

    The parser checks row shape and numeric conversion. Semantic validity, such
    as non-negative class IDs or box size, is handled in ``validation.py``.
    """

    parts = line.strip().split()
    if len(parts) != 5:
        raise AnnotationError(f"expected 5 fields, found {len(parts)}")
    try:
        class_id_float = float(parts[0])
        class_id = int(class_id_float)
        if class_id_float != class_id:
            raise ValueError
        x_center, y_center, width, height = (float(value) for value in parts[1:])
    except ValueError as exc:
        raise AnnotationError("annotation row contains non-numeric values") from exc
    return YoloBox(class_id, x_center, y_center, width, height)


def parse_yolo_text(text: str, stats: ProcessingStats | None = None) -> list[YoloBox]:
    """Parse YOLO text, collecting malformed-row statistics when requested."""

    boxes: list[YoloBox] = []
    if not text.strip() and stats is not None:
        stats.empty_annotation_files += 1
    for raw_line in text.splitlines():
        if not raw_line.strip():
            continue
        try:
            boxes.append(parse_yolo_line(raw_line))
        except AnnotationError:
            if stats is not None:
                stats.malformed_rows += 1
    if stats is not None:
        stats.total_boxes_read += len(boxes)
    return boxes


def read_yolo_file(path: Path, stats: ProcessingStats | None = None) -> list[YoloBox]:
    """Read and parse a YOLO annotation file."""

    return parse_yolo_text(path.read_text(encoding="utf-8"), stats)


def serialize_yolo_box(box: YoloBox, precision: int = 6) -> str:
    """Serialize one YOLO box with fixed decimal precision."""

    values = (box.x_center, box.y_center, box.width, box.height)
    formatted = " ".join(f"{value:.{precision}f}" for value in values)
    return f"{box.class_id} {formatted}"


def serialize_yolo_boxes(boxes: list[YoloBox], precision: int = 6) -> str:
    """Serialize YOLO boxes to text, ending with a newline when non-empty."""

    if not boxes:
        return ""
    return "\n".join(serialize_yolo_box(box, precision) for box in boxes) + "\n"


def write_yolo_file(path: Path, boxes: list[YoloBox], precision: int = 6) -> None:
    """Write YOLO boxes to disk, creating the parent directory."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialize_yolo_boxes(boxes, precision), encoding="utf-8")
