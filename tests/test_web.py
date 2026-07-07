from pathlib import Path

import gradio as gr
import pytest

from yolo_mosaic.services import create_synthetic_dataset
from yolo_mosaic.web import _generate, build_interface


def test_web_generate_wrapper(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    create_synthetic_dataset(dataset, num_images=4, seed=3)
    raw, visualized, annotations, zip_path = _generate(
        sorted((dataset / "images").glob("*.jpg")),
        sorted((dataset / "labels").glob("*.txt")),
        2,
        200,
        200,
        3,
        0.25,
    )
    assert raw.shape == (200, 200, 3)
    assert visualized.shape == (200, 200, 3)
    assert annotations
    assert Path(zip_path).exists()


def test_web_rejects_invalid_grid() -> None:
    with pytest.raises(gr.Error):
        _generate([], [], 4, 200, 200, 1, 0.25)


def test_web_interface_builds() -> None:
    assert build_interface() is not None
