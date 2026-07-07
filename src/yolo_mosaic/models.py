"""Typed data models for the YOLO mosaic toolkit.

The public models are intentionally light on business logic. Geometry,
validation, I/O, and workflow behavior live in their own modules so the same
objects can be reused by tests, CLI commands, services, and the web UI.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

FillPolicy = Literal["repeat", "blank", "error"]
GridSize = Literal[2, 3]
ImageFormat = Literal["jpg", "png"]
MissingLabelPolicy = Literal["empty", "skip", "error"]


@dataclass(frozen=True, slots=True)
class YoloBox:
    """A YOLO normalized bounding box.

    Invariants are enforced by validation functions rather than the constructor
    because malformed input rows still need to be represented and reported.
    """

    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float


@dataclass(frozen=True, slots=True)
class PixelBox:
    """A pixel-space XYXY bounding box."""

    class_id: int
    x_min: float
    y_min: float
    x_max: float
    y_max: float


@dataclass(frozen=True, slots=True)
class LetterboxTransform:
    """Resize and padding parameters shared by image and box transforms."""

    source_width: int
    source_height: int
    target_width: int
    target_height: int
    scale_x: float
    scale_y: float
    pad_left: float
    pad_top: float
    resized_width: int
    resized_height: int


@dataclass(slots=True)
class AnnotatedImage:
    """An image array and its validated pixel boxes."""

    image_path: Path
    label_path: Path | None
    image: NDArray[np.uint8]
    boxes: list[PixelBox]

    @property
    def width(self) -> int:
        """Image width in pixels."""

        return int(self.image.shape[1])

    @property
    def height(self) -> int:
        """Image height in pixels."""

        return int(self.image.shape[0])


@dataclass(frozen=True, slots=True)
class ValidationConfig:
    """Validation thresholds used after repair and clipping."""

    min_box_width: float = 2.0
    min_box_height: float = 2.0
    min_normalized_area: float = 0.0
    min_visible_ratio: float = 0.25


@dataclass(frozen=True, slots=True)
class MosaicConfig:
    """Configuration for deterministic equal-cell mosaic generation."""

    grid_size: GridSize = 2
    output_width: int = 1280
    output_height: int = 1280
    seed: int | None = None
    fill_policy: FillPolicy = "repeat"
    padding_color: tuple[int, int, int] = (114, 114, 114)
    image_format: ImageFormat = "jpg"
    jpeg_quality: int = 95
    overwrite: bool = False


@dataclass(slots=True)
class ProcessingStats:
    """Counters collected while validating, generating, and writing datasets."""

    total_boxes_read: int = 0
    valid_boxes: int = 0
    repaired_boxes: int = 0
    clipped_boxes: int = 0
    rejected_boxes: int = 0
    malformed_rows: int = 0
    empty_annotation_files: int = 0
    missing_annotation_files: int = 0
    skipped_images: int = 0
    generated_mosaics: int = 0

    def merge(self, other: ProcessingStats) -> None:
        """Add another stats object into this one."""

        self.total_boxes_read += other.total_boxes_read
        self.valid_boxes += other.valid_boxes
        self.repaired_boxes += other.repaired_boxes
        self.clipped_boxes += other.clipped_boxes
        self.rejected_boxes += other.rejected_boxes
        self.malformed_rows += other.malformed_rows
        self.empty_annotation_files += other.empty_annotation_files
        self.missing_annotation_files += other.missing_annotation_files
        self.skipped_images += other.skipped_images
        self.generated_mosaics += other.generated_mosaics

    def as_dict(self) -> dict[str, int]:
        """Return a JSON-serializable counter dictionary."""

        return {
            "total_boxes_read": self.total_boxes_read,
            "valid_boxes": self.valid_boxes,
            "repaired_boxes": self.repaired_boxes,
            "clipped_boxes": self.clipped_boxes,
            "rejected_boxes": self.rejected_boxes,
            "malformed_rows": self.malformed_rows,
            "empty_annotation_files": self.empty_annotation_files,
            "missing_annotation_files": self.missing_annotation_files,
            "skipped_images": self.skipped_images,
            "generated_mosaics": self.generated_mosaics,
        }


@dataclass(slots=True)
class MosaicResult:
    """A generated mosaic image with pixel and YOLO annotations."""

    image: NDArray[np.uint8]
    pixel_boxes: list[PixelBox]
    yolo_boxes: list[YoloBox]
    stats: ProcessingStats
