from pathlib import Path

import pytest

from yolo_mosaic.synthetic import SyntheticDataError, generate_synthetic_dataset


def test_synthetic_data_generation(tmp_path: Path) -> None:
    summary = generate_synthetic_dataset(
        tmp_path / "dataset",
        num_images=8,
        seed=123,
        include_invalid=True,
    )
    dataset = tmp_path / "dataset"
    assert summary["num_images"] == 8
    assert (dataset / "images" / "synthetic_000.jpg").exists()
    assert (
        (dataset / "labels" / "synthetic_004.txt")
        .read_text(encoding="utf-8")
        .endswith("malformed row\n")
    )
    assert not (dataset / "labels" / "synthetic_005.txt").exists()
    assert (dataset / "classes.txt").read_text(encoding="utf-8").splitlines() == [
        "square",
        "wide_box",
        "tall_box",
    ]


def test_synthetic_refuses_invalid_counts_and_overwrite(tmp_path: Path) -> None:
    with pytest.raises(SyntheticDataError):
        generate_synthetic_dataset(tmp_path / "dataset", num_images=0)
    generate_synthetic_dataset(tmp_path / "dataset", num_images=1)
    with pytest.raises(SyntheticDataError):
        generate_synthetic_dataset(tmp_path / "dataset", num_images=1)
