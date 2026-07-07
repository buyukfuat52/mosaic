"""Typer command-line interface for the YOLO mosaic toolkit."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, cast

import typer

from yolo_mosaic.models import FillPolicy, GridSize, ImageFormat, MissingLabelPolicy
from yolo_mosaic.services import (
    create_synthetic_dataset,
    generate_mosaics,
    run_benchmark,
    validate_dataset,
    visualize_dataset,
)

app = typer.Typer(help="Generate, validate, visualize, and demo YOLO mosaic annotations.")


def _grid(value: int) -> GridSize:
    if value not in (2, 3):
        raise typer.BadParameter("grid must be 2 or 3")
    return cast(GridSize, value)


def _json_stats(payload: object) -> None:
    typer.echo(json.dumps(payload, indent=2))


@app.command()
def generate(
    images_dir: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    labels_dir: Annotated[Path, typer.Option(file_okay=False)],
    output_images_dir: Annotated[Path, typer.Option(file_okay=False)],
    output_labels_dir: Annotated[Path, typer.Option(file_okay=False)],
    grid: Annotated[int, typer.Option()] = 2,
    width: Annotated[int, typer.Option()] = 1280,
    height: Annotated[int, typer.Option()] = 1280,
    count: Annotated[int, typer.Option()] = 1,
    seed: Annotated[int, typer.Option()] = 42,
    fill_policy: Annotated[FillPolicy, typer.Option()] = "repeat",
    missing_label_policy: Annotated[MissingLabelPolicy, typer.Option()] = "empty",
    min_visible_ratio: Annotated[float, typer.Option()] = 0.25,
    min_box_width: Annotated[float, typer.Option()] = 2.0,
    min_box_height: Annotated[float, typer.Option()] = 2.0,
    min_normalized_area: Annotated[float, typer.Option()] = 0.0,
    image_format: Annotated[ImageFormat, typer.Option()] = "jpg",
    jpeg_quality: Annotated[int, typer.Option()] = 95,
    overwrite: Annotated[bool, typer.Option()] = False,
    manifest_path: Annotated[Path | None, typer.Option()] = None,
    report_path: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """Generate deterministic 2x2 or 3x3 mosaics."""

    result = generate_mosaics(
        images_dir=images_dir,
        labels_dir=labels_dir,
        output_images_dir=output_images_dir,
        output_labels_dir=output_labels_dir,
        grid=_grid(grid),
        width=width,
        height=height,
        count=count,
        seed=seed,
        fill_policy=fill_policy,
        missing_label_policy=missing_label_policy,
        min_visible_ratio=min_visible_ratio,
        min_box_width=min_box_width,
        min_box_height=min_box_height,
        min_normalized_area=min_normalized_area,
        image_format=image_format,
        jpeg_quality=jpeg_quality,
        overwrite=overwrite,
        manifest_path=manifest_path,
        report_path=report_path,
    )
    _json_stats(result.stats.as_dict())


@app.command()
def validate(
    images_dir: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    labels_dir: Annotated[Path, typer.Option(file_okay=False)],
    output_labels_dir: Annotated[Path | None, typer.Option(file_okay=False)] = None,
    missing_label_policy: Annotated[MissingLabelPolicy, typer.Option()] = "empty",
    min_visible_ratio: Annotated[float, typer.Option()] = 0.25,
    min_box_width: Annotated[float, typer.Option()] = 2.0,
    min_box_height: Annotated[float, typer.Option()] = 2.0,
    min_normalized_area: Annotated[float, typer.Option()] = 0.0,
    overwrite: Annotated[bool, typer.Option()] = False,
    report_path: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """Validate a YOLO dataset and optionally write repaired labels."""

    stats = validate_dataset(
        images_dir=images_dir,
        labels_dir=labels_dir,
        output_labels_dir=output_labels_dir,
        missing_label_policy=missing_label_policy,
        min_visible_ratio=min_visible_ratio,
        min_box_width=min_box_width,
        min_box_height=min_box_height,
        overwrite=overwrite,
        report_path=report_path,
        min_normalized_area=min_normalized_area,
    )
    _json_stats(stats.as_dict())


@app.command()
def visualize(
    images_dir: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    labels_dir: Annotated[Path, typer.Option(file_okay=False)],
    output_dir: Annotated[Path, typer.Option(file_okay=False)],
    missing_label_policy: Annotated[MissingLabelPolicy, typer.Option()] = "empty",
    min_visible_ratio: Annotated[float, typer.Option()] = 0.25,
    min_box_width: Annotated[float, typer.Option()] = 2.0,
    min_box_height: Annotated[float, typer.Option()] = 2.0,
    min_normalized_area: Annotated[float, typer.Option()] = 0.0,
    image_format: Annotated[ImageFormat, typer.Option()] = "jpg",
    jpeg_quality: Annotated[int, typer.Option()] = 95,
    overwrite: Annotated[bool, typer.Option()] = False,
) -> None:
    """Render images with repaired bounding boxes."""

    result = visualize_dataset(
        images_dir=images_dir,
        labels_dir=labels_dir,
        output_dir=output_dir,
        missing_label_policy=missing_label_policy,
        min_visible_ratio=min_visible_ratio,
        min_box_width=min_box_width,
        min_box_height=min_box_height,
        image_format=image_format,
        jpeg_quality=jpeg_quality,
        overwrite=overwrite,
        min_normalized_area=min_normalized_area,
    )
    _json_stats(result.stats.as_dict())


@app.command()
def synthetic(
    output_dir: Annotated[Path, typer.Option(file_okay=False)],
    num_images: Annotated[int, typer.Option()] = 30,
    seed: Annotated[int, typer.Option()] = 42,
    include_invalid: Annotated[bool, typer.Option()] = False,
    overwrite: Annotated[bool, typer.Option()] = False,
) -> None:
    """Create deterministic synthetic YOLO data."""

    summary = create_synthetic_dataset(output_dir, num_images, seed, include_invalid, overwrite)
    _json_stats(summary)


@app.command()
def benchmark(
    output_dir: Annotated[Path, typer.Option(file_okay=False)] = Path("examples/benchmark"),
    num_images: Annotated[int, typer.Option()] = 30,
    count: Annotated[int, typer.Option()] = 10,
    seed: Annotated[int, typer.Option()] = 42,
    overwrite: Annotated[bool, typer.Option()] = False,
) -> None:
    """Measure throughput for a synthetic end-to-end workflow."""

    payload = run_benchmark(output_dir, num_images, count, seed, overwrite)
    _json_stats(payload)


@app.command()
def web(
    host: Annotated[str, typer.Option()] = "127.0.0.1",
    port: Annotated[int, typer.Option()] = 7860,
) -> None:
    """Launch the Gradio web interface."""

    from yolo_mosaic.web import launch

    launch(host, port)


if __name__ == "__main__":
    app()
