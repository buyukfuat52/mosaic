"""Bounding-box visualization utilities."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

from yolo_mosaic.image_ops import save_image
from yolo_mosaic.models import ImageFormat, PixelBox


def color_for_class(class_id: int) -> tuple[int, int, int]:
    """Return a deterministic BGR color for a class ID."""

    red = (37 * (class_id + 3)) % 255
    green = (17 * (class_id + 7)) % 255
    blue = (97 * (class_id + 11)) % 255
    return int(blue), int(green), int(red)


def draw_boxes(
    image: NDArray[np.uint8],
    boxes: list[PixelBox],
    class_names: dict[int, str] | None = None,
) -> NDArray[np.uint8]:
    """Draw boxes on a BGR image and return a copy."""

    output = image.copy()
    height, width = output.shape[:2]
    thickness = max(1, int(round(min(width, height) / 300)))
    font_scale = max(0.35, min(width, height) / 900)
    for box in boxes:
        color = color_for_class(box.class_id)
        p1 = int(round(box.x_min)), int(round(box.y_min))
        p2 = int(round(box.x_max)), int(round(box.y_max))
        cv2.rectangle(output, p1, p2, color, thickness)
        label = (
            class_names.get(box.class_id, str(box.class_id)) if class_names else str(box.class_id)
        )
        label_size, baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            thickness,
        )
        label_x = max(0, p1[0])
        label_y = max(label_size[1] + baseline, p1[1])
        cv2.rectangle(
            output,
            (label_x, label_y - label_size[1] - baseline),
            (label_x + label_size[0], label_y + baseline),
            color,
            -1,
        )
        cv2.putText(
            output,
            label,
            (label_x, label_y - baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
    return output


def bgr_to_rgb(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Convert OpenCV BGR image data to RGB for UI display."""

    return cast(NDArray[np.uint8], cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def rgb_to_bgr(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Convert RGB UI image data to OpenCV BGR order."""

    return cast(NDArray[np.uint8], cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def save_visualization(
    path: Path,
    image: NDArray[np.uint8],
    boxes: list[PixelBox],
    class_names: dict[int, str] | None = None,
    image_format: ImageFormat = "jpg",
    jpeg_quality: int = 95,
    overwrite: bool = False,
) -> None:
    """Draw and save a visualization image."""

    save_image(
        path,
        draw_boxes(image, boxes, class_names),
        image_format=image_format,
        jpeg_quality=jpeg_quality,
        overwrite=overwrite,
    )
