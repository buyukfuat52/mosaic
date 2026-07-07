import math

import pytest

from yolo_mosaic.models import PixelBox, ProcessingStats, ValidationConfig
from yolo_mosaic.validation import validate_and_repair_box, validate_and_repair_boxes


def test_valid_box_is_accepted() -> None:
    result = validate_and_repair_box(PixelBox(0, 1.0, 2.0, 20.0, 30.0), 100, 100)
    assert result.accepted
    assert result.box == PixelBox(0, 1.0, 2.0, 20.0, 30.0)
    assert result.visible_ratio == pytest.approx(1.0)


def test_reversed_coordinates_are_repaired() -> None:
    result = validate_and_repair_box(PixelBox(0, 20.0, 30.0, 1.0, 2.0), 100, 100)
    assert result.accepted
    assert result.repaired
    assert result.box == PixelBox(0, 1.0, 2.0, 20.0, 30.0)


def test_out_of_bounds_box_is_clipped() -> None:
    result = validate_and_repair_box(PixelBox(0, -10.0, -10.0, 50.0, 50.0), 100, 100)
    assert result.accepted
    assert result.clipped
    assert result.box == PixelBox(0, 0.0, 0.0, 50.0, 50.0)


def test_non_finite_values_are_rejected() -> None:
    result = validate_and_repair_box(PixelBox(0, math.nan, 0.0, 10.0, 10.0), 100, 100)
    assert not result.accepted
    assert result.reason == "non_finite"


def test_negative_class_id_is_rejected() -> None:
    result = validate_and_repair_box(PixelBox(-1, 0.0, 0.0, 10.0, 10.0), 100, 100)
    assert not result.accepted
    assert result.reason == "negative_class_id"


def test_zero_area_box_is_rejected() -> None:
    result = validate_and_repair_box(PixelBox(0, 10.0, 10.0, 10.0, 20.0), 100, 100)
    assert not result.accepted
    assert result.reason == "zero_area"


def test_min_visible_area_filtering() -> None:
    config = ValidationConfig(min_visible_ratio=0.5)
    result = validate_and_repair_box(PixelBox(0, -90.0, 0.0, 10.0, 10.0), 100, 100, config)
    assert not result.accepted
    assert result.reason == "visible_ratio"
    assert result.visible_ratio == pytest.approx(0.1)


def test_box_clipped_out_of_view_is_rejected() -> None:
    result = validate_and_repair_box(PixelBox(0, -20.0, -20.0, -10.0, -10.0), 100, 100)
    assert not result.accepted
    assert result.reason == "not_visible"


def test_minimum_pixel_width_filtering() -> None:
    config = ValidationConfig(min_box_width=3.0)
    result = validate_and_repair_box(PixelBox(0, 0.0, 0.0, 2.0, 10.0), 100, 100, config)
    assert not result.accepted
    assert result.reason == "min_width"


def test_minimum_pixel_height_filtering() -> None:
    config = ValidationConfig(min_box_height=3.0)
    result = validate_and_repair_box(PixelBox(0, 0.0, 0.0, 10.0, 2.0), 100, 100, config)
    assert not result.accepted
    assert result.reason == "min_height"


def test_minimum_normalized_area_filtering() -> None:
    config = ValidationConfig(min_normalized_area=0.02)
    result = validate_and_repair_box(PixelBox(0, 0.0, 0.0, 10.0, 10.0), 100, 100, config)
    assert not result.accepted
    assert result.reason == "min_normalized_area"


def test_batch_validation_updates_stats() -> None:
    stats = ProcessingStats()
    valid = validate_and_repair_boxes(
        [PixelBox(0, -1.0, 0.0, 10.0, 10.0), PixelBox(0, 5.0, 5.0, 5.0, 9.0)],
        100,
        100,
        stats=stats,
    )
    assert valid == [PixelBox(0, 0.0, 0.0, 10.0, 10.0)]
    assert stats.valid_boxes == 1
    assert stats.clipped_boxes == 1
    assert stats.rejected_boxes == 1
