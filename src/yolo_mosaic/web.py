"""Gradio web interface for one-shot mosaic generation."""

from __future__ import annotations

from typing import cast

import gradio as gr

from yolo_mosaic.models import GridSize
from yolo_mosaic.services import build_web_mosaic, coerce_uploaded_paths


def _generate(
    image_files: object,
    label_files: object,
    grid: int,
    width: int,
    height: int,
    seed: int,
    min_visible_ratio: float,
) -> tuple[object, object, str, str]:
    image_paths = coerce_uploaded_paths(image_files)
    label_paths = coerce_uploaded_paths(label_files)
    if grid not in (2, 3):
        raise gr.Error("Grid must be 2 or 3.")
    raw, visualized, annotations, zip_path = build_web_mosaic(
        image_paths,
        label_paths,
        grid=cast(GridSize, grid),
        width=width,
        height=height,
        seed=seed,
        min_visible_ratio=min_visible_ratio,
    )
    return raw, visualized, annotations, str(zip_path)


def build_interface() -> gr.Blocks:
    """Create the Gradio Blocks interface."""

    with gr.Blocks(title="YOLO Mosaic Annotation Toolkit") as demo:
        gr.Markdown("# YOLO Mosaic Annotation Toolkit")
        with gr.Row():
            image_files = gr.File(file_count="multiple", label="Images")
            label_files = gr.File(file_count="multiple", label="YOLO labels")
        with gr.Row():
            grid = gr.Radio([2, 3], value=2, label="Grid")
            width = gr.Number(value=1280, precision=0, label="Output width")
            height = gr.Number(value=1280, precision=0, label="Output height")
            seed = gr.Number(value=42, precision=0, label="Seed")
            min_visible_ratio = gr.Slider(
                0.0,
                1.0,
                value=0.25,
                step=0.05,
                label="Min visible ratio",
            )
        generate_button = gr.Button("Generate mosaic", variant="primary")
        with gr.Row():
            raw_image = gr.Image(label="Raw mosaic")
            visualized_image = gr.Image(label="Bounding-box visualization")
        annotation_text = gr.Textbox(label="Generated YOLO annotations", lines=8)
        zip_file = gr.File(label="Download ZIP")
        generate_button.click(
            _generate,
            inputs=[image_files, label_files, grid, width, height, seed, min_visible_ratio],
            outputs=[raw_image, visualized_image, annotation_text, zip_file],
        )
    return demo


def launch(host: str = "127.0.0.1", port: int = 7860) -> None:
    """Launch the web application."""

    build_interface().launch(server_name=host, server_port=port)
