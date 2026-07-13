from pathlib import Path
from typing import cast

import numpy as np
import pytest

from yolo_mosaic.models import (
    AnnotatedImage,
    FillPolicy,
    GridSize,
    MosaicConfig,
    PixelBox,
    ValidationConfig,
)
from yolo_mosaic.mosaic import MosaicError, create_mosaic


def _item(
    class_id: int,
    boxes: list[PixelBox] | None = None,
    width: int = 100,
    height: int = 100,
) -> AnnotatedImage:
    image = np.full((height, width, 3), class_id * 20, dtype=np.uint8)
    return AnnotatedImage(
        image_path=Path(f"image_{class_id}.jpg"),
        label_path=None,
        image=image,
        boxes=boxes or [],
    )


def test_2x2_bottom_right_coordinate_mapping() -> None:
    items = [
        _item(0),
        _item(1),
        _item(2),
        _item(3, [PixelBox(7, 0.0, 0.0, 100.0, 100.0)]),
    ]
    result = create_mosaic(items, MosaicConfig(grid_size=2, output_width=200, output_height=200))
    assert result.pixel_boxes == [PixelBox(7, 100.0, 100.0, 200.0, 200.0)]
    assert result.yolo_boxes[0].x_center == pytest.approx(0.75)
    assert result.yolo_boxes[0].y_center == pytest.approx(0.75)
    assert result.yolo_boxes[0].width == pytest.approx(0.5)
    assert result.yolo_boxes[0].height == pytest.approx(0.5)


def test_2x2_letterbox_padding_is_included_in_mapping() -> None:
    items = [
        _item(0, [PixelBox(1, 0.0, 0.0, 200.0, 100.0)], width=200, height=100),
        _item(1),
        _item(2),
        _item(3),
    ]
    result = create_mosaic(items, MosaicConfig(grid_size=2, output_width=200, output_height=200))
    assert result.pixel_boxes[0] == PixelBox(1, 0.0, 25.0, 100.0, 75.0)
    assert result.yolo_boxes[0].x_center == pytest.approx(0.25)
    assert result.yolo_boxes[0].y_center == pytest.approx(0.25)
    assert result.yolo_boxes[0].width == pytest.approx(0.5)
    assert result.yolo_boxes[0].height == pytest.approx(0.25)


def test_repeat_policy_fills_all_cells() -> None:
    result = create_mosaic(
        [_item(0, [PixelBox(2, 0.0, 0.0, 100.0, 100.0)])],
        MosaicConfig(grid_size=2, output_width=200, output_height=200, fill_policy="repeat"),
    )
    centers = [(box.x_center, box.y_center) for box in result.yolo_boxes]
    assert centers == pytest.approx([(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)])


def test_blank_policy_allows_empty_sources() -> None:
    result = create_mosaic(
        [],
        MosaicConfig(grid_size=2, output_width=200, output_height=200, fill_policy="blank"),
    )
    assert result.image.shape == (200, 200, 3)
    assert result.pixel_boxes == []


def test_error_policy_rejects_too_few_sources() -> None:
    with pytest.raises(MosaicError):
        create_mosaic([_item(0)], MosaicConfig(grid_size=2, fill_policy="error"))


def test_mosaic_rejects_invalid_grid_and_dimensions() -> None:
    with pytest.raises(MosaicError):
        create_mosaic(
            [_item(0)],
            MosaicConfig(
                grid_size=cast(GridSize, 4),
                output_width=200,
                output_height=200,
            ),
        )
    with pytest.raises(MosaicError):
        create_mosaic([_item(0)], MosaicConfig(grid_size=2, output_width=0, output_height=200))


def test_mosaic_rejects_unsupported_fill_policy() -> None:
    with pytest.raises(MosaicError):
        create_mosaic(
            [_item(0)],
            MosaicConfig(grid_size=2, fill_policy=cast(FillPolicy, "unsupported")),
        )


def test_mosaic_counts_rejected_transformed_boxes() -> None:
    result = create_mosaic(
        [_item(0, [PixelBox(1, 0.0, 0.0, 1.0, 1.0)])],
        MosaicConfig(grid_size=2, output_width=200, output_height=200, fill_policy="repeat"),
    )
    assert result.pixel_boxes == []
    assert result.stats.rejected_boxes == 4


def test_custom_minimum_box_width_applies_after_downscaling() -> None:
    items = [
        _item(0, [PixelBox(1, 0.0, 0.0, 7.0, 20.0)]),
        _item(1),
        _item(2),
        _item(3),
    ]
    config = MosaicConfig(grid_size=2, output_width=100, output_height=100)

    default_result = create_mosaic(items, config)
    strict_result = create_mosaic(items, config, ValidationConfig(min_box_width=4.0))

    assert default_result.pixel_boxes == [PixelBox(1, 0.0, 0.0, 3.5, 10.0)]
    assert strict_result.pixel_boxes == []
    assert strict_result.stats.rejected_boxes == 1


def test_custom_minimum_box_height_applies_after_downscaling() -> None:
    items = [
        _item(0, [PixelBox(1, 0.0, 0.0, 20.0, 7.0)]),
        _item(1),
        _item(2),
        _item(3),
    ]
    config = MosaicConfig(grid_size=2, output_width=100, output_height=100)

    default_result = create_mosaic(items, config)
    strict_result = create_mosaic(items, config, ValidationConfig(min_box_height=4.0))

    assert default_result.pixel_boxes == [PixelBox(1, 0.0, 0.0, 10.0, 3.5)]
    assert strict_result.pixel_boxes == []
    assert strict_result.stats.rejected_boxes == 1


def test_custom_minimum_visible_ratio_applies_to_transformed_box() -> None:
    items = [
        _item(0, [PixelBox(1, -60.0, 0.0, 40.0, 20.0)]),
        _item(1),
        _item(2),
        _item(3),
    ]
    config = MosaicConfig(grid_size=2, output_width=100, output_height=100)

    default_result = create_mosaic(items, config)
    strict_result = create_mosaic(items, config, ValidationConfig(min_visible_ratio=0.5))

    assert default_result.pixel_boxes == [PixelBox(1, 0.0, 0.0, 20.0, 10.0)]
    assert strict_result.pixel_boxes == []
    assert strict_result.stats.rejected_boxes == 1


def test_default_validation_config_remains_backward_compatible() -> None:
    items = [
        _item(0, [PixelBox(1, 0.0, 0.0, 20.0, 20.0)]),
        _item(1),
        _item(2),
        _item(3),
    ]
    config = MosaicConfig(grid_size=2, output_width=100, output_height=100)

    implicit_default = create_mosaic(items, config)
    explicit_default = create_mosaic(items, config, ValidationConfig())

    assert implicit_default.pixel_boxes == explicit_default.pixel_boxes
    assert implicit_default.yolo_boxes == explicit_default.yolo_boxes
    assert implicit_default.stats == explicit_default.stats
