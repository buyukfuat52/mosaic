"""Shared application workflows used by CLI and web interfaces."""

from __future__ import annotations

import json
import platform
import shutil
import tempfile
import time
import uuid
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import cv2
import numpy as np
from numpy.typing import NDArray

from yolo_mosaic.annotations import parse_yolo_text, serialize_yolo_boxes, write_yolo_file
from yolo_mosaic.dataset import load_annotated_images
from yolo_mosaic.geometry import pixel_to_yolo, yolo_to_pixel
from yolo_mosaic.image_ops import load_image, save_image
from yolo_mosaic.models import (
    AnnotatedImage,
    FillPolicy,
    GridSize,
    ImageFormat,
    MissingLabelPolicy,
    MosaicConfig,
    ProcessingStats,
    ValidationConfig,
)
from yolo_mosaic.mosaic import create_mosaic
from yolo_mosaic.reporting import write_json_report, write_manifest
from yolo_mosaic.synthetic import generate_synthetic_dataset
from yolo_mosaic.validation import validate_and_repair_boxes
from yolo_mosaic.visualization import bgr_to_rgb, draw_boxes, save_visualization


@dataclass(frozen=True, slots=True)
class WorkflowOutputs:
    """Generated file paths and counters for a workflow."""

    stats: ProcessingStats
    images: list[Path]
    labels: list[Path]


class UploadedFileLike(Protocol):
    """Minimal uploaded-file shape used by Gradio file components."""

    name: str


WEB_EXPORT_RETENTION_SECONDS = 6 * 60 * 60
WEB_EXPORT_ROOT = Path(tempfile.gettempdir()) / "yolo_mosaic_web_exports"


@dataclass(frozen=True, slots=True)
class UploadMatchReport:
    """Summary of uploaded image and label matching by filename stem."""

    image_count: int
    label_count: int
    matched_count: int
    images_without_labels: list[str]
    extra_label_files: list[str]
    duplicate_image_stems: list[str]
    duplicate_label_stems: list[str]
    unused_files: list[str]
    expected_cells: int

    @property
    def warning_count(self) -> int:
        """Return the number of warning categories with at least one item."""

        warnings = [
            self.images_without_labels,
            self.extra_label_files,
            self.duplicate_image_stems,
            self.duplicate_label_stems,
        ]
        insufficient = self.image_count < self.expected_cells
        return sum(1 for items in warnings if items) + int(insufficient)

    def as_markdown(self) -> str:
        """Render the report as compact Markdown for the web UI."""

        grid_size = int(self.expected_cells**0.5)
        lines = [
            "### Input summary",
            f"- Uploaded images: {self.image_count}",
            f"- Uploaded annotation files: {self.label_count}",
            f"- Matched image-label pairs: {self.matched_count}",
            f"- Images without labels: {len(self.images_without_labels)}",
            f"- Unused files: {len(self.unused_files)}",
            f"- Expected mosaic cells: {self.expected_cells}",
        ]
        warnings: list[str] = []
        if self.images_without_labels:
            warnings.append(
                "Missing annotation files: " + ", ".join(self.images_without_labels)
            )
        if self.extra_label_files:
            warnings.append("Extra annotation files: " + ", ".join(self.extra_label_files))
        if self.duplicate_image_stems:
            warnings.append("Duplicate image stems: " + ", ".join(self.duplicate_image_stems))
        if self.duplicate_label_stems:
            warnings.append("Duplicate label stems: " + ", ".join(self.duplicate_label_stems))
        if self.image_count < self.expected_cells:
            warnings.append(
                f"Insufficient images for a {grid_size}x{grid_size} grid; "
                "repeat fill will reuse uploads."
            )
        if warnings:
            lines.append("")
            lines.append("### Warnings")
            lines.extend(f"- {warning}" for warning in warnings)
        else:
            lines.extend(["", "No matching warnings."])
        return "\n".join(lines)


def _ensure_output_file(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing file: {path}")


def _paths_by_stem(paths: list[Path]) -> dict[str, list[Path]]:
    grouped: defaultdict[str, list[Path]] = defaultdict(list)
    for path in paths:
        grouped[path.stem].append(path)
    return dict(grouped)


def _duplicate_stems(grouped_paths: dict[str, list[Path]]) -> list[str]:
    return sorted(stem for stem, paths in grouped_paths.items() if len(paths) > 1)


def build_upload_match_report(
    image_paths: list[Path],
    label_paths: list[Path],
    grid: GridSize,
) -> UploadMatchReport:
    """Summarize web uploads matched by filename stem."""

    image_groups = _paths_by_stem(image_paths)
    label_groups = _paths_by_stem(label_paths)
    image_stems = set(image_groups)
    label_stems = set(label_groups)
    missing_label_stems = sorted(image_stems - label_stems)
    extra_label_stems = sorted(label_stems - image_stems)
    duplicate_image_stems = _duplicate_stems(image_groups)
    duplicate_label_stems = _duplicate_stems(label_groups)
    unused_files = [
        path.name
        for stem in extra_label_stems
        for path in sorted(label_groups[stem], key=lambda item: item.name)
    ]
    for stem in duplicate_label_stems:
        duplicate_paths = sorted(label_groups[stem], key=lambda item: item.name)
        unused_files.extend(path.name for path in duplicate_paths[1:])

    return UploadMatchReport(
        image_count=len(image_paths),
        label_count=len(label_paths),
        matched_count=sum(1 for path in image_paths if path.stem in label_stems),
        images_without_labels=[
            path.name
            for stem in missing_label_stems
            for path in sorted(image_groups[stem], key=lambda item: item.name)
        ],
        extra_label_files=[
            path.name
            for stem in extra_label_stems
            for path in sorted(label_groups[stem], key=lambda item: item.name)
        ],
        duplicate_image_stems=duplicate_image_stems,
        duplicate_label_stems=duplicate_label_stems,
        unused_files=sorted(unused_files),
        expected_cells=int(grid) * int(grid),
    )


def parse_class_names_text(text: str) -> dict[int, str]:
    """Parse a YOLO classes.txt-style file into class-id labels."""

    return {
        index: line.strip()
        for index, line in enumerate(text.splitlines())
        if line.strip()
    }


def read_class_names_file(path: Path | None) -> dict[int, str] | None:
    """Read optional class names from a text file."""

    if path is None:
        return None
    return parse_class_names_text(path.read_text(encoding="utf-8"))


def cleanup_web_exports(
    export_root: Path = WEB_EXPORT_ROOT,
    max_age_seconds: int = WEB_EXPORT_RETENTION_SECONDS,
    now: float | None = None,
) -> int:
    """Delete old web export directories while keeping recent downloads available."""

    if not export_root.exists():
        return 0
    active_now = time.time() if now is None else now
    removed = 0
    for child in export_root.iterdir():
        if not child.is_dir():
            continue
        age_seconds = active_now - child.stat().st_mtime
        if age_seconds <= max_age_seconds:
            continue
        shutil.rmtree(child, ignore_errors=True)
        removed += 1
    return removed


def _create_web_export_dir() -> Path:
    cleanup_web_exports()
    WEB_EXPORT_ROOT.mkdir(parents=True, exist_ok=True)
    export_dir = WEB_EXPORT_ROOT / f"{int(time.time())}_{uuid.uuid4().hex}"
    export_dir.mkdir()
    return export_dir


def generate_mosaics(
    images_dir: Path,
    labels_dir: Path,
    output_images_dir: Path,
    output_labels_dir: Path,
    grid: GridSize = 2,
    width: int = 1280,
    height: int = 1280,
    count: int = 1,
    seed: int | None = 42,
    fill_policy: FillPolicy = "repeat",
    missing_label_policy: MissingLabelPolicy = "empty",
    min_visible_ratio: float = 0.25,
    min_box_width: float = 2.0,
    min_box_height: float = 2.0,
    image_format: ImageFormat = "jpg",
    jpeg_quality: int = 95,
    overwrite: bool = False,
    manifest_path: Path | None = None,
    report_path: Path | None = None,
    min_normalized_area: float = 0.0,
) -> WorkflowOutputs:
    """Generate deterministic mosaic images and YOLO label files."""

    if count <= 0:
        raise ValueError("count must be positive")
    stats = ProcessingStats()
    validation_config = ValidationConfig(
        min_box_width=min_box_width,
        min_box_height=min_box_height,
        min_visible_ratio=min_visible_ratio,
        min_normalized_area=min_normalized_area,
    )
    items = load_annotated_images(
        images_dir,
        labels_dir,
        missing_label_policy=missing_label_policy,
        validation_config=validation_config,
        stats=stats,
    )
    if not items and fill_policy != "blank":
        raise ValueError("no usable images found")

    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)
    image_paths: list[Path] = []
    label_paths: list[Path] = []
    manifest_rows: list[dict[str, str]] = []
    extension = "jpg" if image_format == "jpg" else "png"

    for index in range(count):
        mosaic_seed = None if seed is None else seed + index
        config = MosaicConfig(
            grid_size=grid,
            output_width=width,
            output_height=height,
            seed=mosaic_seed,
            fill_policy=fill_policy,
            image_format=image_format,
            jpeg_quality=jpeg_quality,
            overwrite=overwrite,
        )
        result = create_mosaic(items, config, validation_config)
        stats.merge(result.stats)
        image_path = output_images_dir / f"mosaic_{index:04d}.{extension}"
        label_path = output_labels_dir / f"mosaic_{index:04d}.txt"
        _ensure_output_file(image_path, overwrite)
        _ensure_output_file(label_path, overwrite)
        save_image(image_path, result.image, image_format, jpeg_quality, overwrite)
        write_yolo_file(label_path, result.yolo_boxes)
        image_paths.append(image_path)
        label_paths.append(label_path)
        manifest_rows.append(
            {
                "image_path": str(image_path),
                "label_path": str(label_path),
                "boxes": str(len(result.yolo_boxes)),
                "seed": str(mosaic_seed),
            }
        )

    if manifest_path is not None:
        write_manifest(manifest_path, manifest_rows)
    if report_path is not None:
        write_json_report(
            report_path,
            stats,
            {"generated_images": [str(path) for path in image_paths]},
        )
    return WorkflowOutputs(stats, image_paths, label_paths)


def validate_dataset(
    images_dir: Path,
    labels_dir: Path,
    output_labels_dir: Path | None = None,
    missing_label_policy: MissingLabelPolicy = "empty",
    min_visible_ratio: float = 0.25,
    min_box_width: float = 2.0,
    min_box_height: float = 2.0,
    overwrite: bool = False,
    report_path: Path | None = None,
    min_normalized_area: float = 0.0,
) -> ProcessingStats:
    """Validate a dataset and optionally write repaired labels elsewhere."""

    stats = ProcessingStats()
    validation_config = ValidationConfig(
        min_box_width=min_box_width,
        min_box_height=min_box_height,
        min_visible_ratio=min_visible_ratio,
        min_normalized_area=min_normalized_area,
    )
    items = load_annotated_images(
        images_dir,
        labels_dir,
        missing_label_policy=missing_label_policy,
        validation_config=validation_config,
        stats=stats,
    )
    if output_labels_dir is not None:
        output_labels_dir.mkdir(parents=True, exist_ok=True)
        for item in items:
            label_path = output_labels_dir / f"{item.image_path.stem}.txt"
            _ensure_output_file(label_path, overwrite)
            yolo_boxes = [pixel_to_yolo(box, item.width, item.height) for box in item.boxes]
            write_yolo_file(label_path, yolo_boxes)
    if report_path is not None:
        write_json_report(report_path, stats)
    return stats


def visualize_dataset(
    images_dir: Path,
    labels_dir: Path,
    output_dir: Path,
    missing_label_policy: MissingLabelPolicy = "empty",
    min_visible_ratio: float = 0.25,
    min_box_width: float = 2.0,
    min_box_height: float = 2.0,
    image_format: ImageFormat = "jpg",
    jpeg_quality: int = 95,
    overwrite: bool = False,
    min_normalized_area: float = 0.0,
) -> WorkflowOutputs:
    """Render bounding-box preview images for a dataset."""

    stats = ProcessingStats()
    validation_config = ValidationConfig(
        min_box_width=min_box_width,
        min_box_height=min_box_height,
        min_visible_ratio=min_visible_ratio,
        min_normalized_area=min_normalized_area,
    )
    items = load_annotated_images(
        images_dir,
        labels_dir,
        missing_label_policy=missing_label_policy,
        validation_config=validation_config,
        stats=stats,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    extension = "jpg" if image_format == "jpg" else "png"
    images: list[Path] = []
    for item in items:
        output_path = output_dir / f"{item.image_path.stem}.{extension}"
        _ensure_output_file(output_path, overwrite)
        save_visualization(
            output_path,
            item.image,
            item.boxes,
            image_format=image_format,
            jpeg_quality=jpeg_quality,
            overwrite=overwrite,
        )
        images.append(output_path)
    return WorkflowOutputs(stats, images, [])


def create_synthetic_dataset(
    output_dir: Path,
    num_images: int = 30,
    seed: int = 42,
    include_invalid: bool = False,
    overwrite: bool = False,
) -> dict[str, object]:
    """Service wrapper for synthetic dataset generation."""

    return generate_synthetic_dataset(output_dir, num_images, seed, include_invalid, overwrite)


def build_web_mosaic(
    image_paths: list[Path],
    label_paths: list[Path],
    grid: GridSize = 2,
    width: int = 1280,
    height: int = 1280,
    seed: int = 42,
    min_visible_ratio: float = 0.25,
    min_box_width: float = 2.0,
    min_box_height: float = 2.0,
    min_normalized_area: float = 0.0,
    class_names: dict[int, str] | None = None,
    display_class_names: bool = False,
) -> tuple[NDArray[np.uint8], NDArray[np.uint8], str, Path]:
    """Build one mosaic for the Gradio web workflow and export a ZIP."""

    if not image_paths:
        raise ValueError("upload at least one image")
    if width <= 0 or height <= 0:
        raise ValueError("output width and height must be positive")

    label_groups = _paths_by_stem(label_paths)
    label_map = {
        stem: sorted(paths, key=lambda path: path.name)[0]
        for stem, paths in label_groups.items()
    }
    validation_config = ValidationConfig(
        min_visible_ratio=min_visible_ratio,
        min_box_width=min_box_width,
        min_box_height=min_box_height,
        min_normalized_area=min_normalized_area,
    )
    items: list[AnnotatedImage] = []
    for image_path in image_paths:
        image = load_image(image_path)
        h, w = image.shape[:2]
        stats = ProcessingStats()
        label_path = label_map.get(image_path.stem)
        yolo_boxes = (
            parse_yolo_text(label_path.read_text(encoding="utf-8"), stats)
            if label_path
            else []
        )
        pixel_boxes = [yolo_to_pixel(box, int(w), int(h)) for box in yolo_boxes]
        boxes = validate_and_repair_boxes(pixel_boxes, int(w), int(h), validation_config, stats)
        items.append(AnnotatedImage(image_path, label_path, image, boxes))

    result = create_mosaic(
        items,
        MosaicConfig(
            grid_size=grid,
            output_width=width,
            output_height=height,
            seed=seed,
            fill_policy="repeat",
        ),
        validation_config,
    )
    annotation_text = serialize_yolo_boxes(result.yolo_boxes)
    visualized = draw_boxes(
        result.image,
        result.pixel_boxes,
        class_names if display_class_names else None,
    )

    temp_dir = _create_web_export_dir()
    raw_path = temp_dir / "mosaic.jpg"
    vis_path = temp_dir / "mosaic_visualized.jpg"
    label_path = temp_dir / "mosaic.txt"
    zip_path = temp_dir / "mosaic_outputs.zip"
    save_image(raw_path, result.image, overwrite=True)
    save_image(vis_path, visualized, overwrite=True)
    label_path.write_text(annotation_text, encoding="utf-8")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(raw_path, arcname="mosaic.jpg")
        archive.write(vis_path, arcname="mosaic_visualized.jpg")
        archive.write(label_path, arcname="mosaic.txt")

    return bgr_to_rgb(result.image), bgr_to_rgb(visualized), annotation_text, zip_path


def run_benchmark(
    output_dir: Path,
    num_images: int = 30,
    count: int = 10,
    seed: int = 42,
    overwrite: bool = False,
) -> dict[str, object]:
    """Measure synthetic parsing, loading, mosaic generation, and visualization."""

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = output_dir / "benchmark_dataset"
    mosaics_dir = output_dir / "benchmark_outputs" / "images"
    labels_dir = output_dir / "benchmark_outputs" / "labels"
    visualized_dir = output_dir / "benchmark_outputs" / "visualized"

    start = time.perf_counter()
    create_synthetic_dataset(dataset_dir, num_images, seed, overwrite=overwrite)
    synthetic_seconds = time.perf_counter() - start

    start = time.perf_counter()
    generate_result = generate_mosaics(
        dataset_dir / "images",
        dataset_dir / "labels",
        mosaics_dir,
        labels_dir,
        grid=2,
        count=count,
        seed=seed,
        overwrite=overwrite,
    )
    generate_seconds = time.perf_counter() - start

    start = time.perf_counter()
    visualize_result = visualize_dataset(
        mosaics_dir,
        labels_dir,
        visualized_dir,
        overwrite=overwrite,
    )
    visualize_seconds = time.perf_counter() - start

    payload: dict[str, object] = {
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "opencv": cv2.__version__,
            "numpy": np.__version__,
        },
        "parameters": {
            "num_images": num_images,
            "count": count,
            "seed": seed,
        },
        "timings_seconds": {
            "synthetic_generation": synthetic_seconds,
            "mosaic_generation": generate_seconds,
            "visualization": visualize_seconds,
        },
        "measured_rates": {
            "mosaics_per_second": count / generate_seconds if generate_seconds > 0 else None,
            "visualizations_per_second": len(visualize_result.images) / visualize_seconds
            if visualize_seconds > 0
            else None,
        },
        "stats": generate_result.stats.as_dict(),
    }
    (output_dir / "benchmark_report.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    return payload


def coerce_uploaded_paths(files: object) -> list[Path]:
    """Convert Gradio file objects or strings to paths."""

    if files is None:
        return []
    if not isinstance(files, list):
        files = [files]
    paths: list[Path] = []
    for item in files:
        if isinstance(item, Path):
            paths.append(item)
            continue
        if isinstance(item, str):
            paths.append(Path(item))
            continue
        uploaded = cast(UploadedFileLike, item)
        paths.append(Path(uploaded.name))
    return paths
