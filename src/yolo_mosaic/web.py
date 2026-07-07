"""Gradio web interface for one-shot mosaic generation."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import gradio as gr

from yolo_mosaic.models import GridSize
from yolo_mosaic.services import (
    build_upload_match_report,
    build_web_mosaic,
    coerce_uploaded_paths,
    read_class_names_file,
)

MIN_OUTPUT_SIZE = 64
MAX_OUTPUT_SIZE = 4096
DEFAULT_PROGRESS = gr.Progress()


def _coerce_grid(grid: int) -> GridSize:
    if grid not in (2, 3):
        raise gr.Error("Grid must be 2 or 3.")
    return cast(GridSize, grid)


def _coerce_output_size(value: int | float, label: str) -> int:
    size = int(value)
    if size < MIN_OUTPUT_SIZE or size > MAX_OUTPUT_SIZE:
        raise gr.Error(f"{label} must be between {MIN_OUTPUT_SIZE} and {MAX_OUTPUT_SIZE} pixels.")
    return size


def _status_for_report(warning_count: int) -> str:
    if warning_count:
        return f"Ready with {warning_count} matching warning(s)."
    return "Ready."


def _review_uploads(
    image_files: object,
    label_files: object,
    grid: int,
) -> tuple[str, str]:
    grid_value = _coerce_grid(grid)
    image_paths = coerce_uploaded_paths(image_files)
    label_paths = coerce_uploaded_paths(label_files)
    report = build_upload_match_report(image_paths, label_paths, grid_value)
    return report.as_markdown(), _status_for_report(report.warning_count)


def _example_dataset_paths(grid: int) -> tuple[list[str], list[str], str | None, str, str]:
    grid_value = _coerce_grid(grid)
    dataset_dir = Path("examples/synthetic_dataset")
    image_paths = sorted((dataset_dir / "images").glob("*.jpg"))
    label_paths = sorted((dataset_dir / "labels").glob("*.txt"))
    class_names_path = dataset_dir / "classes.txt"
    if len(image_paths) < int(grid_value) * int(grid_value):
        raise gr.Error(
            "Example assets are missing. Run `yolo-mosaic synthetic "
            "--output-dir examples/synthetic_dataset --num-images 30 --seed 42` first."
        )
    selected_images = image_paths[: int(grid_value) * int(grid_value)]
    selected_stems = {path.stem for path in selected_images}
    selected_labels = [path for path in label_paths if path.stem in selected_stems]
    report = build_upload_match_report(selected_images, selected_labels, grid_value)
    return (
        [str(path) for path in selected_images],
        [str(path) for path in selected_labels],
        str(class_names_path) if class_names_path.exists() else None,
        report.as_markdown(),
        "Loaded repository example files.",
    )


def _generate(
    image_files: object,
    label_files: object,
    class_names_file: object,
    display_class_names: bool,
    grid: int,
    width: int | float,
    height: int | float,
    seed: int,
    min_visible_ratio: float,
    min_box_width: float,
    min_box_height: float,
    min_normalized_area: float,
    progress: gr.Progress = DEFAULT_PROGRESS,
) -> tuple[object, object, str, str, str, str]:
    try:
        progress(0.1, desc="Matching uploads")
        grid_value = _coerce_grid(grid)
        width_value = _coerce_output_size(width, "Output width")
        height_value = _coerce_output_size(height, "Output height")
        image_paths = coerce_uploaded_paths(image_files)
        label_paths = coerce_uploaded_paths(label_files)
        class_name_paths = coerce_uploaded_paths(class_names_file)
        class_names = read_class_names_file(class_name_paths[0] if class_name_paths else None)
        report = build_upload_match_report(image_paths, label_paths, grid_value)

        progress(0.35, desc="Parsing annotations")
        progress(0.65, desc="Generating mosaic")
        raw, visualized, annotations, zip_path = build_web_mosaic(
            image_paths,
            label_paths,
            grid=grid_value,
            width=width_value,
            height=height_value,
            seed=int(seed),
            min_visible_ratio=min_visible_ratio,
            min_box_width=min_box_width,
            min_box_height=min_box_height,
            min_normalized_area=min_normalized_area,
            class_names=class_names,
            display_class_names=display_class_names,
        )
        progress(0.9, desc="Exporting ZIP")
        status = (
            f"Generated {grid_value}x{grid_value} mosaic with "
            f"{len(annotations.splitlines())} annotation row(s)."
        )
        return raw, visualized, annotations, str(zip_path), report.as_markdown(), status
    except gr.Error:
        raise
    except (OSError, RuntimeError, ValueError) as exc:
        raise gr.Error(str(exc)) from exc


def build_interface() -> gr.Blocks:
    """Create the Gradio Blocks interface."""

    with gr.Blocks(title="YOLO Mosaic Annotation Toolkit") as demo:
        gr.Markdown("# YOLO Mosaic Annotation Toolkit")
        with gr.Row():
            image_files = gr.File(
                file_count="multiple",
                file_types=["image"],
                label="Images",
            )
            label_files = gr.File(
                file_count="multiple",
                file_types=[".txt"],
                label="YOLO labels",
            )
            class_names_file = gr.File(
                file_count="single",
                file_types=[".txt"],
                label="Class names",
            )
        with gr.Row():
            grid = gr.Radio(
                [2, 3],
                value=2,
                label="Grid",
                info="2 uses four cells; 3 uses nine cells.",
            )
            width = gr.Number(
                value=1280,
                precision=0,
                minimum=MIN_OUTPUT_SIZE,
                maximum=MAX_OUTPUT_SIZE,
                label="Output width",
            )
            height = gr.Number(
                value=1280,
                precision=0,
                minimum=MIN_OUTPUT_SIZE,
                maximum=MAX_OUTPUT_SIZE,
                label="Output height",
            )
            seed = gr.Number(
                value=42,
                precision=0,
                label="Seed",
                info="Same seed and inputs produce the same mosaic.",
            )
        with gr.Row():
            min_visible_ratio = gr.Slider(
                0.0,
                1.0,
                value=0.25,
                step=0.05,
                label="Min visible ratio",
                info="Reject boxes that keep too little area after clipping.",
            )
            min_box_width = gr.Number(
                value=2.0,
                minimum=0.0,
                maximum=4096.0,
                step=0.5,
                label="Min box width",
            )
            min_box_height = gr.Number(
                value=2.0,
                minimum=0.0,
                maximum=4096.0,
                step=0.5,
                label="Min box height",
            )
            min_normalized_area = gr.Number(
                value=0.0,
                minimum=0.0,
                maximum=1.0,
                step=0.001,
                label="Min normalized area",
            )
            display_class_names = gr.Checkbox(value=True, label="Display class names")
        with gr.Row():
            review_button = gr.Button("Review files")
            example_button = gr.Button("Load demo files")
            generate_button = gr.Button("Generate mosaic", variant="primary")
        match_report = gr.Markdown("### Input summary\n- Uploaded images: 0")
        status_text = gr.Textbox(label="Status", value="Waiting for uploads.", interactive=False)
        with gr.Row():
            raw_image = gr.Image(label="Raw mosaic")
            visualized_image = gr.Image(label="Bounding-box visualization")
        annotation_text = gr.Textbox(label="Generated YOLO annotations", lines=8)
        zip_file = gr.File(label="Download ZIP")
        gr.ClearButton(
            value="Clear",
            components=[
                image_files,
                label_files,
                class_names_file,
                raw_image,
                visualized_image,
                annotation_text,
                zip_file,
                match_report,
                status_text,
            ],
        )
        review_button.click(
            _review_uploads,
            inputs=[image_files, label_files, grid],
            outputs=[match_report, status_text],
        )
        example_button.click(
            _example_dataset_paths,
            inputs=[grid],
            outputs=[image_files, label_files, class_names_file, match_report, status_text],
        )
        generate_button.click(
            _generate,
            inputs=[
                image_files,
                label_files,
                class_names_file,
                display_class_names,
                grid,
                width,
                height,
                seed,
                min_visible_ratio,
                min_box_width,
                min_box_height,
                min_normalized_area,
            ],
            outputs=[
                raw_image,
                visualized_image,
                annotation_text,
                zip_file,
                match_report,
                status_text,
            ],
        )
    return demo


def launch(host: str = "127.0.0.1", port: int = 7860) -> None:
    """Launch the web application."""

    build_interface().launch(server_name=host, server_port=port)
