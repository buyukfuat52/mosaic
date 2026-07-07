import pytest

from yolo_mosaic.geometry import (
    apply_letterbox_transform,
    box_area,
    calculate_letterbox_transform,
    clip_box_to_bounds,
    pixel_to_yolo,
    reorder_box_coordinates,
    transform_pixel_box,
    yolo_to_pixel,
)
from yolo_mosaic.models import PixelBox, YoloBox


def test_yolo_to_pixel_conversion() -> None:
    box = yolo_to_pixel(YoloBox(3, 0.5, 0.5, 0.25, 0.5), 400, 200)
    assert box == PixelBox(3, 150.0, 50.0, 250.0, 150.0)


def test_invalid_dimensions_are_rejected() -> None:
    with pytest.raises(ValueError):
        yolo_to_pixel(YoloBox(0, 0.5, 0.5, 0.2, 0.2), 0, 100)


def test_pixel_to_yolo_conversion() -> None:
    yolo = pixel_to_yolo(PixelBox(2, 10.0, 20.0, 30.0, 60.0), 100, 200)
    assert yolo == YoloBox(2, 0.2, 0.2, 0.2, 0.2)


def test_round_trip_conversion_tolerance() -> None:
    original = YoloBox(1, 0.333333, 0.666667, 0.123456, 0.234567)
    pixel = yolo_to_pixel(original, 640, 480)
    round_trip = pixel_to_yolo(pixel, 640, 480)
    assert round_trip.x_center == pytest.approx(original.x_center)
    assert round_trip.y_center == pytest.approx(original.y_center)
    assert round_trip.width == pytest.approx(original.width)
    assert round_trip.height == pytest.approx(original.height)


def test_boundary_clipping() -> None:
    clipped = clip_box_to_bounds(PixelBox(0, -5.0, 10.0, 120.0, 130.0), 100, 80)
    assert clipped == PixelBox(0, 0.0, 10.0, 100.0, 80.0)


def test_reversed_coordinate_repair() -> None:
    repaired = reorder_box_coordinates(PixelBox(0, 50.0, 40.0, 10.0, 5.0))
    assert repaired == PixelBox(0, 10.0, 5.0, 50.0, 40.0)


def test_box_area_ignores_negative_extents() -> None:
    assert box_area(PixelBox(0, 10.0, 10.0, 5.0, 20.0)) == 0.0


def test_letterbox_scale_and_padding() -> None:
    transform = calculate_letterbox_transform(200, 100, 100, 100)
    assert transform.scale_x == pytest.approx(0.5)
    assert transform.scale_y == pytest.approx(0.5)
    assert transform.resized_width == 100
    assert transform.resized_height == 50
    assert transform.pad_left == pytest.approx(0.0)
    assert transform.pad_top == pytest.approx(25.0)


def test_apply_letterbox_transform() -> None:
    transform = calculate_letterbox_transform(200, 100, 100, 100)
    transformed = apply_letterbox_transform(PixelBox(0, 0.0, 0.0, 200.0, 100.0), transform)
    assert transformed == PixelBox(0, 0.0, 25.0, 100.0, 75.0)


def test_transform_pixel_box_with_cell_offset() -> None:
    transformed = transform_pixel_box(PixelBox(1, 0.0, 0.0, 10.0, 20.0), 2.0, 3.0, 5.0, 7.0)
    assert transformed == PixelBox(1, 5.0, 7.0, 25.0, 67.0)
