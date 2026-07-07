# YOLO Mosaic Annotation Toolkit

A production-oriented Python toolkit for generating deterministic 2x2 and 3x3 YOLO object-detection mosaics while preserving, repairing, validating, visualizing, and exporting annotations.

## Problem Statement

YOLO mosaic augmentation is useful only when bounding boxes are transformed with the exact same resize, padding, and placement parameters used for the image. This project keeps that geometry centralized and tested so CLI, web, and service workflows all produce consistent labels.

## Features

- Parse and serialize YOLO normalized annotations.
- Convert between YOLO normalized and pixel XYXY boxes.
- Repair reversed coordinates and clip out-of-bounds boxes.
- Reject malformed rows, non-finite values, negative class IDs, zero-area boxes, and boxes below configurable thresholds.
- Generate deterministic 2x2 and 3x3 equal-cell mosaics with letterbox placement.
- Render deterministic bounding-box previews.
- Generate reproducible synthetic datasets with optional invalid examples.
- Run shared workflows from Typer CLI and Gradio web UI.
- Export web outputs as a ZIP archive.
- Run linting, type checking, coverage tests, CI, Docker, and benchmark commands.

## Engineering Highlights

- Geometry, validation, I/O, mosaic generation, services, CLI, and web UI are separate modules.
- Bounding-box math is implemented in `src/yolo_mosaic/geometry.py` and covered by numerical tests.
- `LetterboxTransform` is shared between image resizing and box transformation.
- Randomness uses local seeded generators for reproducibility.
- Output files are not overwritten unless `--overwrite` is supplied.

## Architecture Overview

See `docs/architecture.md` for a module diagram and design notes.

Core flow:

```text
dataset -> annotations -> geometry -> validation -> mosaic -> services -> CLI/web
```

## Installation

```bash
python -m pip install -e ".[dev]"
```

Python 3.11 or newer is required.

## Quick Start

```bash
yolo-mosaic synthetic --output-dir examples/synthetic_dataset --num-images 30 --seed 42
yolo-mosaic generate --images-dir examples/synthetic_dataset/images --labels-dir examples/synthetic_dataset/labels --output-images-dir examples/outputs/images --output-labels-dir examples/outputs/labels --grid 2 --count 2 --seed 42
yolo-mosaic visualize --images-dir examples/outputs/images --labels-dir examples/outputs/labels --output-dir examples/outputs/visualized
```

## Dataset Layout

```text
dataset/
  images/
    image_001.jpg
    image_002.png
  labels/
    image_001.txt
    image_002.txt
```

Labels use one YOLO row per object:

```text
<class_id> <x_center> <y_center> <width> <height>
```

## CLI Examples

```bash
yolo-mosaic generate \
  --images-dir dataset/images \
  --labels-dir dataset/labels \
  --output-images-dir output/images \
  --output-labels-dir output/labels \
  --grid 2 \
  --width 1280 \
  --height 1280 \
  --count 100 \
  --seed 42 \
  --fill-policy repeat \
  --missing-label-policy empty \
  --min-visible-ratio 0.25
```

```bash
yolo-mosaic validate --images-dir dataset/images --labels-dir dataset/labels --output-labels-dir repaired/labels
yolo-mosaic visualize --images-dir output/images --labels-dir output/labels --output-dir output/visualized
yolo-mosaic benchmark --output-dir examples/benchmark --overwrite
```

## Web UI

```bash
yolo-mosaic web --host 127.0.0.1 --port 7860
```

The Gradio interface accepts multiple images and YOLO label files, lets you configure grid size, dimensions, seed, and visibility threshold, then displays the raw mosaic, visualization, generated annotation text, and a ZIP download.

## Synthetic Data

```bash
yolo-mosaic synthetic --output-dir examples/synthetic_dataset --num-images 30 --seed 42 --include-invalid
```

The generator creates varied image dimensions, geometric shapes, at least three classes, empty annotations, optional malformed examples, `classes.txt`, and `summary.json`.

## Bounding-Box Mathematics

YOLO normalized to pixel XYXY:

```text
x_min = (x_center - width / 2) * W
y_min = (y_center - height / 2) * H
x_max = (x_center + width / 2) * W
y_max = (y_center + height / 2) * H
```

Pixel XYXY to YOLO normalized:

```text
x_center = ((x_min + x_max) / 2) / W
y_center = ((y_min + y_max) / 2) / H
width    = (x_max - x_min) / W
height   = (y_max - y_min) / H
```

For each mosaic cell:

```text
new_x_min = old_x_min * scale_x + offset_x
new_y_min = old_y_min * scale_y + offset_y
new_x_max = old_x_max * scale_x + offset_x
new_y_max = old_y_max * scale_y + offset_y
```

Letterbox mode uses equal `scale_x` and `scale_y`, plus cell offset and top-left padding.

## Validation and Repair Policies

- Reorder reversed coordinates where possible.
- Clip boxes to image boundaries.
- Reject non-finite coordinates.
- Reject negative class IDs.
- Reject zero-area boxes.
- Reject boxes below minimum pixel width or height.
- Reject boxes below minimum normalized area.
- Reject boxes whose clipped visible-area ratio is below the configured threshold.

Defaults are minimum visible ratio `0.25` and minimum box size `2x2 px`.

## Testing

```bash
python -m ruff check .
python -m mypy src
python -m pytest --cov=src/yolo_mosaic --cov-report=term-missing
```

## Benchmarking

```bash
yolo-mosaic benchmark --output-dir examples/benchmark --overwrite
```

The benchmark prints environment information and measured timings for synthetic generation, mosaic generation, and visualization.

## Docker

```bash
docker build -t yolo-mosaic-toolkit .
docker run --rm yolo-mosaic-toolkit --help
docker run --rm -p 7860:7860 yolo-mosaic-toolkit web --host 0.0.0.0 --port 7860
```

## CI

GitHub Actions prints the installed tool versions, then runs Ruff, mypy, and pytest with
coverage on Python 3.11 and 3.12.

## Example Outputs

Reproducible examples are documented in `docs/examples.md`. Generated files are written under:

- `examples/synthetic_dataset/`
- `examples/outputs/images/`
- `examples/outputs/labels/`
- `examples/outputs/visualized/`

## Reproducibility

Using the same input files, seed, grid size, output dimensions, and validation thresholds produces the same selected source images and transformed annotations.

## Limitations

- The implemented mosaic mode is deterministic equal-cell placement, not randomized YOLO-style mosaic center cropping.
- COCO and Pascal VOC import/export are not included.
- Parallel workers are reserved for future scaling work.

## Roadmap

- Randomized mosaic center with crop-aware visibility filtering.
- COCO and Pascal VOC adapters.
- Dataset statistics dashboard.
- Parallel processing for large datasets.
- Property-based tests for geometry invariants.

## Troubleshooting

- If OpenCV cannot import on Linux, install `libgl1` and `libglib2.0-0`.
- If output files already exist, rerun with `--overwrite` or choose a new output directory.
- If labels are missing, choose `--missing-label-policy empty`, `skip`, or `error`.
- If too few images exist for a grid, choose `--fill-policy repeat`, `blank`, or `error`.

## Contributing

See `CONTRIBUTING.md`.

## License

MIT. See `LICENSE`.
