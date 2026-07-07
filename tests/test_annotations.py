import pytest

from yolo_mosaic.annotations import (
    AnnotationError,
    parse_yolo_line,
    parse_yolo_text,
    serialize_yolo_boxes,
)
from yolo_mosaic.models import ProcessingStats, YoloBox


def test_parse_yolo_line() -> None:
    box = parse_yolo_line("2 0.5 0.25 0.1 0.2")
    assert box == YoloBox(2, 0.5, 0.25, 0.1, 0.2)


def test_parse_yolo_line_rejects_bad_shape_and_class() -> None:
    with pytest.raises(AnnotationError):
        parse_yolo_line("0 0.5")
    with pytest.raises(AnnotationError):
        parse_yolo_line("1.5 0.5 0.5 0.2 0.2")


def test_parse_yolo_text_reports_malformed_rows() -> None:
    stats = ProcessingStats()
    boxes = parse_yolo_text("0 0.5 0.5 0.2 0.2\nbroken\n1 0 0 0 0\n", stats)
    assert boxes == [YoloBox(0, 0.5, 0.5, 0.2, 0.2), YoloBox(1, 0.0, 0.0, 0.0, 0.0)]
    assert stats.malformed_rows == 1
    assert stats.total_boxes_read == 2


def test_empty_annotation_file_is_reported() -> None:
    stats = ProcessingStats()
    assert parse_yolo_text("\n", stats) == []
    assert stats.empty_annotation_files == 1


def test_serialization_precision() -> None:
    text = serialize_yolo_boxes([YoloBox(1, 1 / 3, 0.5, 0.25, 0.125)], precision=4)
    assert text == "1 0.3333 0.5000 0.2500 0.1250\n"
