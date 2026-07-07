"""Deterministic synthetic YOLO dataset generation."""

from __future__ import annotations

import json
import random
from pathlib import Path

import cv2
import numpy as np

from yolo_mosaic.annotations import serialize_yolo_boxes
from yolo_mosaic.geometry import pixel_to_yolo
from yolo_mosaic.image_ops import save_image
from yolo_mosaic.models import PixelBox


class SyntheticDataError(RuntimeError):
    """Raised when synthetic data generation would overwrite existing data."""


def _planned_paths(output_dir: Path, num_images: int) -> list[Path]:
    image_paths = [
        output_dir / "images" / f"synthetic_{index:03d}.jpg" for index in range(num_images)
    ]
    label_paths = [
        output_dir / "labels" / f"synthetic_{index:03d}.txt" for index in range(num_images)
    ]
    return [*image_paths, *label_paths, output_dir / "classes.txt", output_dir / "summary.json"]


def _ensure_no_overwrite(output_dir: Path, num_images: int, overwrite: bool) -> None:
    if overwrite:
        return
    existing = [path for path in _planned_paths(output_dir, num_images) if path.exists()]
    if existing:
        raise SyntheticDataError(f"refusing to overwrite existing synthetic file: {existing[0]}")


def generate_synthetic_dataset(
    output_dir: Path,
    num_images: int = 30,
    seed: int = 42,
    include_invalid: bool = False,
    overwrite: bool = False,
) -> dict[str, object]:
    """Create a deterministic synthetic object-detection dataset."""

    if num_images <= 0:
        raise SyntheticDataError("num_images must be positive")
    _ensure_no_overwrite(output_dir, num_images, overwrite)
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    dimensions = [(320, 240), (640, 480), (480, 360), (300, 500), (512, 512)]
    classes = ["square", "wide_box", "tall_box"]
    images_without_annotations = 0
    missing_annotation_files = 0
    malformed_rows = 0

    for index in range(num_images):
        width, height = dimensions[index % len(dimensions)]
        image = np.full((height, width, 3), 225, dtype=np.uint8)
        boxes: list[PixelBox] = []
        if index % 7 == 0:
            images_without_annotations += 1
        else:
            shape_count = 1 + (index % 3)
            for shape_index in range(shape_count):
                class_id = (index + shape_index) % len(classes)
                box_width = rng.randint(max(20, width // 8), max(25, width // 3))
                box_height = rng.randint(max(20, height // 8), max(25, height // 3))
                x_min = rng.randint(0, max(0, width - box_width - 1))
                y_min = rng.randint(0, max(0, height - box_height - 1))
                x_max = x_min + box_width
                y_max = y_min + box_height
                color = [(30, 120, 220), (70, 180, 70), (220, 80, 80)][class_id]
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, -1)
                boxes.append(
                    PixelBox(class_id, float(x_min), float(y_min), float(x_max), float(y_max))
                )

        image_path = images_dir / f"synthetic_{index:03d}.jpg"
        label_path = labels_dir / f"synthetic_{index:03d}.txt"
        save_image(image_path, image, overwrite=overwrite)

        if include_invalid and index == 5:
            missing_annotation_files += 1
            continue

        yolo_boxes = [pixel_to_yolo(box, width, height) for box in boxes]
        rows = serialize_yolo_boxes(yolo_boxes)
        if include_invalid and index == 1:
            rows += "0 1.20 0.50 0.40 0.40\n"
        if include_invalid and index == 2:
            rows += "1 0.50 0.50 -0.20 0.20\n"
        if include_invalid and index == 3:
            rows += "2 0.50 0.50 0.00 0.00\n"
        if include_invalid and index == 4:
            rows += "malformed row\n"
            malformed_rows += 1
        label_path.write_text(rows, encoding="utf-8")

    (output_dir / "classes.txt").write_text("\n".join(classes) + "\n", encoding="utf-8")
    summary: dict[str, object] = {
        "num_images": num_images,
        "seed": seed,
        "include_invalid": include_invalid,
        "classes": classes,
        "images_without_annotations": images_without_annotations,
        "missing_annotation_files": missing_annotation_files,
        "malformed_rows": malformed_rows,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
