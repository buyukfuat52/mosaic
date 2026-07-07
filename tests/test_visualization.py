from pathlib import Path

import numpy as np
import pytest

from yolo_mosaic.image_ops import ImageOperationError, load_image, save_image
from yolo_mosaic.models import PixelBox
from yolo_mosaic.visualization import bgr_to_rgb, draw_boxes, rgb_to_bgr


def test_visualization_output_dimensions() -> None:
    image = np.zeros((100, 120, 3), dtype=np.uint8)
    rendered = draw_boxes(image, [PixelBox(1, 10.0, 20.0, 50.0, 70.0)])
    assert rendered.shape == image.shape
    assert rendered.dtype == image.dtype
    assert np.any(rendered != image)


def test_visualization_handles_images_with_no_boxes() -> None:
    image = np.zeros((40, 50, 3), dtype=np.uint8)
    rendered = draw_boxes(image, [])
    assert rendered.shape == image.shape
    assert np.array_equal(rendered, image)


def test_bgr_to_rgb_conversion() -> None:
    image = np.array([[[1, 2, 3]]], dtype=np.uint8)
    assert bgr_to_rgb(image).tolist() == [[[3, 2, 1]]]
    assert rgb_to_bgr(bgr_to_rgb(image)).tolist() == image.tolist()


def test_image_io_error_paths(tmp_path: Path) -> None:
    with pytest.raises(ImageOperationError):
        load_image(tmp_path / "missing.jpg")
    path = tmp_path / "image.jpg"
    image = np.zeros((5, 5, 3), dtype=np.uint8)
    save_image(path, image)
    with pytest.raises(FileExistsError):
        save_image(path, image)
