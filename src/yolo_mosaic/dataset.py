"""Dataset discovery and image-label loading."""

from __future__ import annotations

from pathlib import Path

from yolo_mosaic.annotations import read_yolo_file
from yolo_mosaic.geometry import yolo_to_pixel
from yolo_mosaic.image_ops import load_image
from yolo_mosaic.models import (
    AnnotatedImage,
    MissingLabelPolicy,
    PixelBox,
    ProcessingStats,
    ValidationConfig,
)
from yolo_mosaic.validation import validate_and_repair_boxes

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class DatasetError(RuntimeError):
    """Raised when dataset layout or policy prevents processing."""


def iter_image_paths(images_dir: Path) -> list[Path]:
    """Return supported image paths sorted by filename."""

    if not images_dir.exists():
        raise DatasetError(f"images directory does not exist: {images_dir}")
    if not images_dir.is_dir():
        raise DatasetError(f"images path is not a directory: {images_dir}")
    return sorted(
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )


def load_annotation_for_image(
    image_path: Path,
    labels_dir: Path,
    image_width: int,
    image_height: int,
    missing_label_policy: MissingLabelPolicy,
    validation_config: ValidationConfig,
    stats: ProcessingStats,
) -> tuple[Path | None, list[PixelBox], bool]:
    """Load, convert, and validate one image's annotation file.

    Returns ``(label_path, boxes, should_skip_image)``.
    """

    label_path = labels_dir / f"{image_path.stem}.txt"
    if not label_path.exists():
        stats.missing_annotation_files += 1
        if missing_label_policy == "error":
            raise DatasetError(f"missing label file for image: {image_path.name}")
        if missing_label_policy == "skip":
            stats.skipped_images += 1
            return None, [], True
        return None, [], False

    yolo_boxes = read_yolo_file(label_path, stats)
    pixel_boxes = [yolo_to_pixel(box, image_width, image_height) for box in yolo_boxes]
    boxes = validate_and_repair_boxes(
        pixel_boxes,
        image_width,
        image_height,
        validation_config,
        stats,
    )
    return label_path, boxes, False


def load_annotated_images(
    images_dir: Path,
    labels_dir: Path,
    missing_label_policy: MissingLabelPolicy = "empty",
    validation_config: ValidationConfig | None = None,
    stats: ProcessingStats | None = None,
) -> list[AnnotatedImage]:
    """Load all supported images and their repaired annotations."""

    active_stats = stats or ProcessingStats()
    config = validation_config or ValidationConfig()
    images: list[AnnotatedImage] = []
    for image_path in iter_image_paths(images_dir):
        image = load_image(image_path)
        image_height, image_width = image.shape[:2]
        label_path, boxes, skip = load_annotation_for_image(
            image_path,
            labels_dir,
            int(image_width),
            int(image_height),
            missing_label_policy,
            config,
            active_stats,
        )
        if skip:
            continue
        images.append(AnnotatedImage(image_path, label_path, image, boxes))
    return images
