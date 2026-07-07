from pathlib import Path

import numpy as np
import pytest

from yolo_mosaic.models import AnnotatedImage, MosaicConfig, PixelBox
from yolo_mosaic.mosaic import create_mosaic


def _item(class_id: int, boxes: list[PixelBox] | None = None) -> AnnotatedImage:
    image = np.full((90, 90, 3), class_id * 10, dtype=np.uint8)
    return AnnotatedImage(Path(f"image_{class_id}.jpg"), None, image, boxes or [])


def test_3x3_center_and_corner_coordinate_mapping() -> None:
    items = [_item(index) for index in range(9)]
    items[4] = _item(4, [PixelBox(4, 0.0, 0.0, 90.0, 90.0)])
    items[8] = _item(8, [PixelBox(8, 0.0, 0.0, 90.0, 90.0)])
    result = create_mosaic(items, MosaicConfig(grid_size=3, output_width=300, output_height=300))
    assert result.pixel_boxes == [
        PixelBox(4, 100.0, 100.0, 200.0, 200.0),
        PixelBox(8, 200.0, 200.0, 300.0, 300.0),
    ]
    assert result.yolo_boxes[0].x_center == pytest.approx(0.5)
    assert result.yolo_boxes[0].y_center == pytest.approx(0.5)
    assert result.yolo_boxes[0].width == pytest.approx(1 / 3)
    assert result.yolo_boxes[1].x_center == pytest.approx(5 / 6)
    assert result.yolo_boxes[1].y_center == pytest.approx(5 / 6)


def test_seeded_sampling_is_deterministic() -> None:
    items = [
        _item(index, [PixelBox(index, 0.0, 0.0, 90.0, 90.0)])
        for index in range(12)
    ]
    config = MosaicConfig(grid_size=3, output_width=300, output_height=300, seed=123)
    first = create_mosaic(items, config)
    second = create_mosaic(items, config)
    assert first.yolo_boxes == second.yolo_boxes
    assert [box.class_id for box in first.pixel_boxes] == [0, 4, 1, 6, 10, 11, 8, 3, 2]
