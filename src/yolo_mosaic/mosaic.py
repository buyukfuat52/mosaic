"""Deterministic 2x2 and 3x3 mosaic generation."""

from __future__ import annotations

import random
from collections.abc import Sequence

import numpy as np

from yolo_mosaic.geometry import pixel_to_yolo, transform_pixel_box
from yolo_mosaic.image_ops import letterbox_image
from yolo_mosaic.models import (
    AnnotatedImage,
    MosaicConfig,
    MosaicResult,
    PixelBox,
    ProcessingStats,
    ValidationConfig,
)
from yolo_mosaic.validation import validate_and_repair_box


class MosaicError(ValueError):
    """Raised when a mosaic cannot be generated from the provided inputs."""


def _cell_bounds(
    row: int,
    col: int,
    grid_size: int,
    output_width: int,
    output_height: int,
) -> tuple[int, int, int, int]:
    x0 = round(col * output_width / grid_size)
    x1 = round((col + 1) * output_width / grid_size)
    y0 = round(row * output_height / grid_size)
    y1 = round((row + 1) * output_height / grid_size)
    return x0, y0, x1, y1


def _select_items(
    items: Sequence[AnnotatedImage],
    cells: int,
    fill_policy: str,
    rng: random.Random,
) -> list[AnnotatedImage | None]:
    if not items:
        if fill_policy == "blank":
            return [None] * cells
        raise MosaicError("at least one image is required unless fill policy is blank")
    if len(items) >= cells:
        if len(items) == cells:
            return list(items)
        return rng.sample(list(items), cells)
    if fill_policy == "error":
        raise MosaicError(f"need {cells} images for the selected grid, found {len(items)}")
    if fill_policy == "blank":
        return list(items) + [None] * (cells - len(items))
    if fill_policy == "repeat":
        selected: list[AnnotatedImage | None] = []
        while len(selected) < cells:
            selected.extend(items)
        return selected[:cells]
    raise MosaicError(f"unsupported fill policy: {fill_policy}")


def create_mosaic(
    items: Sequence[AnnotatedImage],
    config: MosaicConfig,
    validation_config: ValidationConfig | None = None,
) -> MosaicResult:
    """Generate one deterministic equal-cell mosaic."""

    active_validation_config = validation_config or ValidationConfig()
    if config.grid_size not in (2, 3):
        raise MosaicError("grid size must be 2 or 3")
    if config.output_width <= 0 or config.output_height <= 0:
        raise MosaicError("output dimensions must be positive")

    rng = random.Random(config.seed)
    grid_size = int(config.grid_size)
    cells = grid_size * grid_size
    selected = _select_items(items, cells, config.fill_policy, rng)
    stats = ProcessingStats(generated_mosaics=1)
    canvas = np.full(
        (config.output_height, config.output_width, 3),
        config.padding_color,
        dtype=np.uint8,
    )
    output_boxes: list[PixelBox] = []

    for index, item in enumerate(selected):
        row = index // grid_size
        col = index % grid_size
        cell_x0, cell_y0, cell_x1, cell_y1 = _cell_bounds(
            row,
            col,
            grid_size,
            config.output_width,
            config.output_height,
        )
        cell_width = cell_x1 - cell_x0
        cell_height = cell_y1 - cell_y0
        if item is None:
            continue

        cell_image, transform = letterbox_image(
            item.image,
            cell_width,
            cell_height,
            config.padding_color,
        )
        canvas[cell_y0:cell_y1, cell_x0:cell_x1] = cell_image

        for box in item.boxes:
            transformed = transform_pixel_box(
                box,
                scale_x=transform.scale_x,
                scale_y=transform.scale_y,
                offset_x=cell_x0 + transform.pad_left,
                offset_y=cell_y0 + transform.pad_top,
            )
            result = validate_and_repair_box(
                transformed,
                config.output_width,
                config.output_height,
                active_validation_config,
            )
            if result.accepted and result.box is not None:
                output_boxes.append(result.box)
                stats.valid_boxes += 1
                if result.repaired:
                    stats.repaired_boxes += 1
                if result.clipped:
                    stats.clipped_boxes += 1
            else:
                stats.rejected_boxes += 1
                if result.repaired:
                    stats.repaired_boxes += 1
                if result.clipped:
                    stats.clipped_boxes += 1

    yolo_boxes = [
        pixel_to_yolo(box, config.output_width, config.output_height) for box in output_boxes
    ]
    return MosaicResult(canvas, output_boxes, yolo_boxes, stats)
