"""Bounding-box validation, repair, clipping, and filtering."""

from __future__ import annotations

from dataclasses import dataclass

from yolo_mosaic.geometry import (
    box_area,
    box_height,
    box_width,
    boxes_equal,
    clip_box_to_bounds,
    has_finite_coordinates,
    reorder_box_coordinates,
    require_positive_dimensions,
)
from yolo_mosaic.models import PixelBox, ProcessingStats, ValidationConfig


@dataclass(frozen=True, slots=True)
class BoxValidationResult:
    """The outcome of validating one pixel box."""

    box: PixelBox | None
    accepted: bool
    repaired: bool = False
    clipped: bool = False
    reason: str | None = None
    visible_ratio: float = 0.0


def validate_class_id(class_id: int) -> bool:
    """Return true for non-negative integer class IDs."""

    return isinstance(class_id, int) and class_id >= 0


def validate_and_repair_box(
    box: PixelBox,
    image_width: int,
    image_height: int,
    config: ValidationConfig | None = None,
) -> BoxValidationResult:
    """Repair, clip, and filter one pixel-space box.

    Reversed coordinates are repaired before clipping. Visible-area ratio is
    computed as clipped area divided by the repaired pre-clip area.
    """

    validation_config = config or ValidationConfig()
    require_positive_dimensions(image_width, image_height)

    if not validate_class_id(box.class_id):
        return BoxValidationResult(None, accepted=False, reason="negative_class_id")
    if not has_finite_coordinates(box):
        return BoxValidationResult(None, accepted=False, reason="non_finite")

    repaired_box = reorder_box_coordinates(box)
    repaired = not boxes_equal(box, repaired_box)
    original_area = box_area(repaired_box)
    if original_area <= 0:
        return BoxValidationResult(None, accepted=False, repaired=repaired, reason="zero_area")

    clipped_box = clip_box_to_bounds(repaired_box, image_width, image_height)
    clipped = not boxes_equal(repaired_box, clipped_box)
    clipped_area = box_area(clipped_box)
    visible_ratio = clipped_area / original_area if original_area > 0 else 0.0

    if clipped_area <= 0:
        return BoxValidationResult(
            None,
            accepted=False,
            repaired=repaired,
            clipped=clipped,
            reason="not_visible",
            visible_ratio=visible_ratio,
        )
    if visible_ratio < validation_config.min_visible_ratio:
        return BoxValidationResult(
            None,
            accepted=False,
            repaired=repaired,
            clipped=clipped,
            reason="visible_ratio",
            visible_ratio=visible_ratio,
        )
    if box_width(clipped_box) < validation_config.min_box_width:
        return BoxValidationResult(
            None,
            accepted=False,
            repaired=repaired,
            clipped=clipped,
            reason="min_width",
            visible_ratio=visible_ratio,
        )
    if box_height(clipped_box) < validation_config.min_box_height:
        return BoxValidationResult(
            None,
            accepted=False,
            repaired=repaired,
            clipped=clipped,
            reason="min_height",
            visible_ratio=visible_ratio,
        )
    normalized_area = clipped_area / (image_width * image_height)
    if normalized_area < validation_config.min_normalized_area:
        return BoxValidationResult(
            None,
            accepted=False,
            repaired=repaired,
            clipped=clipped,
            reason="min_normalized_area",
            visible_ratio=visible_ratio,
        )

    return BoxValidationResult(
        clipped_box,
        accepted=True,
        repaired=repaired,
        clipped=clipped,
        visible_ratio=visible_ratio,
    )


def validate_and_repair_boxes(
    boxes: list[PixelBox],
    image_width: int,
    image_height: int,
    config: ValidationConfig | None = None,
    stats: ProcessingStats | None = None,
) -> list[PixelBox]:
    """Validate many boxes and update optional processing statistics."""

    valid_boxes: list[PixelBox] = []
    for box in boxes:
        result = validate_and_repair_box(box, image_width, image_height, config)
        if result.accepted and result.box is not None:
            valid_boxes.append(result.box)
            if stats is not None:
                stats.valid_boxes += 1
                if result.repaired:
                    stats.repaired_boxes += 1
                if result.clipped:
                    stats.clipped_boxes += 1
        elif stats is not None:
            stats.rejected_boxes += 1
            if result.repaired:
                stats.repaired_boxes += 1
            if result.clipped:
                stats.clipped_boxes += 1
    return valid_boxes
