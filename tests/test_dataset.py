from pathlib import Path

import numpy as np
import pytest

from yolo_mosaic.annotations import serialize_yolo_boxes
from yolo_mosaic.dataset import DatasetError, iter_image_paths, load_annotated_images
from yolo_mosaic.geometry import pixel_to_yolo
from yolo_mosaic.image_ops import save_image
from yolo_mosaic.models import PixelBox, ProcessingStats


def _write_image(path: Path) -> None:
    save_image(path, np.zeros((20, 20, 3), dtype=np.uint8), overwrite=True)


def test_missing_label_policy_empty(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()
    _write_image(images_dir / "sample.jpg")
    stats = ProcessingStats()
    items = load_annotated_images(images_dir, labels_dir, "empty", stats=stats)
    assert len(items) == 1
    assert items[0].boxes == []
    assert stats.missing_annotation_files == 1


def test_iter_image_paths_rejects_invalid_directories(tmp_path: Path) -> None:
    with pytest.raises(DatasetError):
        iter_image_paths(tmp_path / "missing")
    file_path = tmp_path / "not_a_directory.txt"
    file_path.write_text("", encoding="utf-8")
    with pytest.raises(DatasetError):
        iter_image_paths(file_path)


def test_missing_label_policy_skip(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()
    _write_image(images_dir / "sample.jpg")
    stats = ProcessingStats()
    assert load_annotated_images(images_dir, labels_dir, "skip", stats=stats) == []
    assert stats.skipped_images == 1


def test_missing_label_policy_error(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()
    _write_image(images_dir / "sample.jpg")
    with pytest.raises(DatasetError):
        load_annotated_images(images_dir, labels_dir, "error")


def test_dataset_loads_and_repairs_labels(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()
    _write_image(images_dir / "sample.jpg")
    labels_dir.joinpath("sample.txt").write_text(
        serialize_yolo_boxes([pixel_to_yolo(PixelBox(0, -5.0, 0.0, 10.0, 10.0), 20, 20)]),
        encoding="utf-8",
    )
    stats = ProcessingStats()
    items = load_annotated_images(images_dir, labels_dir, stats=stats)
    assert items[0].boxes == [PixelBox(0, 0.0, 0.0, 10.0, 10.0)]
    assert stats.clipped_boxes == 1
