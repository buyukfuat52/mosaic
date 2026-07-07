"""Centralized bounding-box geometry and letterbox calculations."""

from __future__ import annotations

import math

from yolo_mosaic.models import LetterboxTransform, PixelBox, YoloBox


class GeometryError(ValueError):
    """Raised when geometry inputs are impossible to transform."""


def require_positive_dimensions(width: int, height: int) -> None:
    """Reject non-positive image dimensions."""

    if width <= 0 or height <= 0:
        raise GeometryError("image dimensions must be positive")


def yolo_to_pixel(box: YoloBox, image_width: int, image_height: int) -> PixelBox:
    """Convert a YOLO normalized box to pixel-space XYXY coordinates."""

    require_positive_dimensions(image_width, image_height)
    x_min = (box.x_center - box.width / 2.0) * image_width
    y_min = (box.y_center - box.height / 2.0) * image_height
    x_max = (box.x_center + box.width / 2.0) * image_width
    y_max = (box.y_center + box.height / 2.0) * image_height
    return PixelBox(box.class_id, x_min, y_min, x_max, y_max)


def pixel_to_yolo(box: PixelBox, image_width: int, image_height: int) -> YoloBox:
    """Convert a pixel-space XYXY box to YOLO normalized coordinates."""

    require_positive_dimensions(image_width, image_height)
    x_center = ((box.x_min + box.x_max) / 2.0) / image_width
    y_center = ((box.y_min + box.y_max) / 2.0) / image_height
    width = (box.x_max - box.x_min) / image_width
    height = (box.y_max - box.y_min) / image_height
    return YoloBox(box.class_id, x_center, y_center, width, height)


def box_width(box: PixelBox) -> float:
    """Return pixel box width."""

    return box.x_max - box.x_min


def box_height(box: PixelBox) -> float:
    """Return pixel box height."""

    return box.y_max - box.y_min


def box_area(box: PixelBox) -> float:
    """Return area for a valid XYXY box, or zero for degenerate boxes."""

    return max(0.0, box_width(box)) * max(0.0, box_height(box))


def has_finite_coordinates(box: PixelBox) -> bool:
    """Return true when all box coordinates and class ID are finite."""

    return (
        math.isfinite(float(box.class_id))
        and math.isfinite(box.x_min)
        and math.isfinite(box.y_min)
        and math.isfinite(box.x_max)
        and math.isfinite(box.y_max)
    )


def reorder_box_coordinates(box: PixelBox) -> PixelBox:
    """Return a box with ``x_min <= x_max`` and ``y_min <= y_max``."""

    return PixelBox(
        box.class_id,
        min(box.x_min, box.x_max),
        min(box.y_min, box.y_max),
        max(box.x_min, box.x_max),
        max(box.y_min, box.y_max),
    )


def clip_box_to_bounds(box: PixelBox, image_width: int, image_height: int) -> PixelBox:
    """Clip a pixel box to inclusive image bounds ``[0, width]`` and ``[0, height]``."""

    require_positive_dimensions(image_width, image_height)
    return PixelBox(
        box.class_id,
        min(max(box.x_min, 0.0), float(image_width)),
        min(max(box.y_min, 0.0), float(image_height)),
        min(max(box.x_max, 0.0), float(image_width)),
        min(max(box.y_max, 0.0), float(image_height)),
    )


def boxes_equal(a: PixelBox, b: PixelBox, tolerance: float = 1e-9) -> bool:
    """Compare pixel coordinates with tolerance while requiring the same class ID."""

    return (
        a.class_id == b.class_id
        and abs(a.x_min - b.x_min) <= tolerance
        and abs(a.y_min - b.y_min) <= tolerance
        and abs(a.x_max - b.x_max) <= tolerance
        and abs(a.y_max - b.y_max) <= tolerance
    )


def transform_pixel_box(
    box: PixelBox,
    scale_x: float,
    scale_y: float,
    offset_x: float,
    offset_y: float,
) -> PixelBox:
    """Scale and translate a pixel-space box."""

    return PixelBox(
        box.class_id,
        box.x_min * scale_x + offset_x,
        box.y_min * scale_y + offset_y,
        box.x_max * scale_x + offset_x,
        box.y_max * scale_y + offset_y,
    )


def calculate_letterbox_transform(
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
) -> LetterboxTransform:
    """Calculate aspect-ratio-preserving resize and centered padding."""

    require_positive_dimensions(source_width, source_height)
    require_positive_dimensions(target_width, target_height)
    scale = min(target_width / source_width, target_height / source_height)
    resized_width = int(round(source_width * scale))
    resized_height = int(round(source_height * scale))
    pad_left = float((target_width - resized_width) // 2)
    pad_top = float((target_height - resized_height) // 2)
    return LetterboxTransform(
        source_width=source_width,
        source_height=source_height,
        target_width=target_width,
        target_height=target_height,
        scale_x=scale,
        scale_y=scale,
        pad_left=pad_left,
        pad_top=pad_top,
        resized_width=resized_width,
        resized_height=resized_height,
    )


def apply_letterbox_transform(box: PixelBox, transform: LetterboxTransform) -> PixelBox:
    """Transform a source pixel box into its letterboxed target coordinates."""

    return transform_pixel_box(
        box,
        scale_x=transform.scale_x,
        scale_y=transform.scale_y,
        offset_x=transform.pad_left,
        offset_y=transform.pad_top,
    )
