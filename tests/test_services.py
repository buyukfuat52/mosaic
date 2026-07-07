import os
from pathlib import Path
from zipfile import ZipFile

from yolo_mosaic.services import (
    build_upload_match_report,
    build_web_mosaic,
    cleanup_web_exports,
    coerce_uploaded_paths,
    create_synthetic_dataset,
    generate_mosaics,
    parse_class_names_text,
    run_benchmark,
    validate_dataset,
    visualize_dataset,
)


def test_service_layer_integration(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    create_synthetic_dataset(dataset, num_images=9, seed=42)
    generated = generate_mosaics(
        dataset / "images",
        dataset / "labels",
        tmp_path / "out" / "images",
        tmp_path / "out" / "labels",
        grid=2,
        width=200,
        height=200,
        count=2,
        seed=42,
        manifest_path=tmp_path / "manifest.csv",
        report_path=tmp_path / "report.json",
    )
    assert len(generated.images) == 2
    assert generated.images[0].exists()
    assert generated.labels[0].read_text(encoding="utf-8")
    assert (tmp_path / "manifest.csv").exists()
    assert (tmp_path / "report.json").exists()

    repaired_stats = validate_dataset(
        dataset / "images",
        dataset / "labels",
        output_labels_dir=tmp_path / "repaired",
    )
    assert repaired_stats.total_boxes_read > 0
    assert (tmp_path / "repaired" / "synthetic_001.txt").exists()

    visualized = visualize_dataset(
        tmp_path / "out" / "images",
        tmp_path / "out" / "labels",
        tmp_path / "visualized",
    )
    assert visualized.images[0].exists()


def test_web_workflow_exports_zip(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    create_synthetic_dataset(dataset, num_images=4, seed=7)
    image_paths = sorted((dataset / "images").glob("*.jpg"))
    label_paths = sorted((dataset / "labels").glob("*.txt"))
    raw, visualized, annotations, zip_path = build_web_mosaic(
        image_paths,
        label_paths,
        grid=2,
        width=200,
        height=200,
        seed=7,
    )
    assert raw.shape == (200, 200, 3)
    assert visualized.shape == (200, 200, 3)
    assert annotations
    with ZipFile(zip_path) as archive:
        assert sorted(archive.namelist()) == ["mosaic.jpg", "mosaic.txt", "mosaic_visualized.jpg"]


def test_benchmark_workflow_and_uploaded_path_coercion(tmp_path: Path) -> None:
    payload = run_benchmark(tmp_path / "benchmark", num_images=4, count=1, seed=5)
    assert payload["parameters"] == {"num_images": 4, "count": 1, "seed": 5}
    assert (tmp_path / "benchmark" / "benchmark_report.json").exists()

    class Uploaded:
        def __init__(self, name: str) -> None:
            self.name = name

    assert coerce_uploaded_paths(None) == []
    assert coerce_uploaded_paths(Uploaded("a.jpg")) == [Path("a.jpg")]
    assert coerce_uploaded_paths(["b.jpg"]) == [Path("b.jpg")]


def test_upload_match_report_flags_missing_extra_and_duplicate_files() -> None:
    image_paths = [
        Path("cat.jpg"),
        Path("dog.jpg"),
        Path("dog.png"),
    ]
    label_paths = [
        Path("cat.txt"),
        Path("extra.txt"),
        Path("dog.txt"),
        Path("dog.copy.txt"),
    ]
    report = build_upload_match_report(image_paths, label_paths, grid=2)
    assert report.image_count == 3
    assert report.label_count == 4
    assert report.matched_count == 3
    assert report.images_without_labels == []
    assert report.extra_label_files == ["dog.copy.txt", "extra.txt"]
    assert report.duplicate_image_stems == ["dog"]
    assert report.duplicate_label_stems == []
    assert report.unused_files == ["dog.copy.txt", "extra.txt"]
    assert "Insufficient images for a 2x2 grid" in report.as_markdown()


def test_class_name_parsing_and_web_export_cleanup(tmp_path: Path) -> None:
    assert parse_class_names_text("person\ncar\n\ntraffic light\n") == {
        0: "person",
        1: "car",
        3: "traffic light",
    }

    export_root = tmp_path / "exports"
    old_dir = export_root / "old"
    new_dir = export_root / "new"
    old_dir.mkdir(parents=True)
    new_dir.mkdir()
    old_time = 100.0
    new_time = 1_000.0
    old_dir.touch()
    new_dir.touch()

    os.utime(old_dir, (old_time, old_time))
    os.utime(new_dir, (new_time, new_time))
    removed = cleanup_web_exports(export_root, max_age_seconds=100, now=1_000.0)
    assert removed == 1
    assert not old_dir.exists()
    assert new_dir.exists()
