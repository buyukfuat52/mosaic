"""Image loading, writing, and letterbox resizing utilities."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

from yolo_mosaic.geometry import calculate_letterbox_transform
from yolo_mosaic.models import ImageFormat, LetterboxTransform


class ImageOperationError(RuntimeError):
    """Raised when image I/O or resizing fails."""


def load_image(path: Path) -> NDArray[np.uint8]:
    """Load an image with OpenCV in BGR channel order."""

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ImageOperationError(f"failed to read image: {path}")
    return cast(NDArray[np.uint8], image)


def save_image(
    path: Path,
    image: NDArray[np.uint8],
    image_format: ImageFormat = "jpg",
    jpeg_quality: int = 95,
    overwrite: bool = False,
) -> None:
    """Write an image to disk without overwriting by default."""

    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing image: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    params: list[int] = []
    if image_format == "jpg":
        params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
    success = cv2.imwrite(str(path), image, params)
    if not success:
        raise ImageOperationError(f"failed to write image: {path}")


def letterbox_image(
    image: NDArray[np.uint8],
    target_width: int,
    target_height: int,
    padding_color: tuple[int, int, int] = (114, 114, 114),
) -> tuple[NDArray[np.uint8], LetterboxTransform]:
    """Resize an image with preserved aspect ratio and centered padding.

    OpenCV arrays are BGR. The padding color is therefore interpreted as BGR in
    file/CLI workflows; UI conversion to RGB happens in the visualization/web
    layer.
    """

    source_height, source_width = image.shape[:2]
    transform = calculate_letterbox_transform(
        source_width=source_width,
        source_height=source_height,
        target_width=target_width,
        target_height=target_height,
    )
    resized = cv2.resize(
        image,
        (transform.resized_width, transform.resized_height),
        interpolation=cv2.INTER_LINEAR,
    )
    canvas = np.full((target_height, target_width, 3), padding_color, dtype=np.uint8)
    left = int(transform.pad_left)
    top = int(transform.pad_top)
    canvas[top : top + transform.resized_height, left : left + transform.resized_width] = resized
    return canvas, transform
