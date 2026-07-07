from pathlib import Path

import gradio as gr
import pytest

from yolo_mosaic.services import create_synthetic_dataset
from yolo_mosaic.web import _generate, _review_uploads, build_interface


def test_web_generate_wrapper(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    create_synthetic_dataset(dataset, num_images=4, seed=3)
    raw, visualized, annotations, zip_path, match_report, status = _generate(
        sorted((dataset / "images").glob("*.jpg")),
        sorted((dataset / "labels").glob("*.txt")),
        dataset / "classes.txt",
        True,
        2,
        200,
        200,
        3,
        0.25,
        2.0,
        2.0,
        0.0,
    )
    assert raw.shape == (200, 200, 3)
    assert visualized.shape == (200, 200, 3)
    assert annotations
    assert Path(zip_path).exists()
    assert "Uploaded images: 4" in match_report
    assert status.startswith("Generated 2x2 mosaic")


def test_web_review_uploads_reports_matching_summary(tmp_path: Path) -> None:
    image = tmp_path / "image_001.jpg"
    label = tmp_path / "image_001.txt"
    extra = tmp_path / "extra.txt"
    report, status = _review_uploads([image], [label, extra], 2)
    assert "Uploaded images: 1" in report
    assert "Matched image-label pairs: 1" in report
    assert "Extra annotation files: extra.txt" in report
    assert "Insufficient images for a 2x2 grid" in report
    assert status == "Ready with 2 matching warning(s)."


def test_web_rejects_invalid_grid() -> None:
    with pytest.raises(gr.Error):
        _generate([], [], None, True, 4, 200, 200, 1, 0.25, 2.0, 2.0, 0.0)


def test_web_interface_builds() -> None:
    assert build_interface() is not None
