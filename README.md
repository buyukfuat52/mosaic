# YOLO Mosaic Annotation Toolkit

Generate deterministic 2x2 and 3x3 YOLO training mosaics with validated bounding-box transformations, a Typer CLI, a Gradio UI, tests, CI, and Docker support.

[![CI](https://github.com/buyukfuat52/mosaic/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/buyukfuat52/mosaic/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)
![License](https://img.shields.io/github/license/buyukfuat52/mosaic)

## Demo

<p align="center">
  <img src="docs/assets/web-interface.png"
       alt="YOLO Mosaic Annotation Toolkit Gradio interface showing upload controls, matching summary, status, and generated mosaic previews"
       width="850">
</p>

<p align="center">
  <img src="docs/assets/mosaic-2x2-visualized.jpg"
       alt="Generated 2x2 YOLO mosaic with transformed bounding-box annotations"
       width="390">
  <img src="docs/assets/mosaic-3x3-visualized.jpg"
       alt="Generated 3x3 YOLO mosaic with transformed bounding-box annotations"
       width="390">
</p>

## Key Features

- Parse and serialize YOLO normalized annotations.
- Convert between YOLO normalized boxes and pixel XYXY boxes.
- Repair reversed coordinates and clip out-of-bounds boxes.
- Reject malformed rows, non-finite values, negative class IDs, zero-area boxes, small boxes, and low-visibility boxes.
- Generate deterministic 2x2 and 3x3 equal-cell mosaics with letterbox placement.
- Render deterministic bounding-box previews with optional class names.
- Generate reproducible synthetic datasets with optional invalid examples.
- Run shared workflows from Typer CLI and Gradio web UI.
- Export web outputs as a ZIP archive.

## Engineering Highlights

- Geometry, validation, I/O, mosaic generation, services, CLI, and web UI are separate modules.
- Bounding-box math lives in `src/yolo_mosaic/geometry.py` and is covered by numerical tests.
- `ValidationConfig` is applied both before and after mosaic transformations.
- `LetterboxTransform` is shared between image resizing and box transformation.
- Randomness uses local seeded generators for reproducibility.
- Output files are not overwritten unless `--overwrite` is supplied.

## Quick Start

Python 3.11 or newer is required.

```bash
python -m pip install -e ".[dev]"
yolo-mosaic synthetic --output-dir examples/synthetic_dataset --num-images 30 --seed 42 --overwrite
yolo-mosaic generate --images-dir examples/synthetic_dataset/images --labels-dir examples/synthetic_dataset/labels --output-images-dir examples/outputs/2x2/images --output-labels-dir examples/outputs/2x2/labels --grid 2 --count 2 --seed 42
yolo-mosaic visualize --images-dir examples/outputs/2x2/images --labels-dir examples/outputs/2x2/labels --output-dir examples/outputs/2x2/visualized
```

## CLI Examples

Generate mosaics:

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
  --min-visible-ratio 0.25 \
  --min-box-width 2 \
  --min-box-height 2 \
  --min-normalized-area 0
```

Validate and repair annotations:

```bash
yolo-mosaic validate \
  --images-dir dataset/images \
  --labels-dir dataset/labels \
  --output-labels-dir repaired/labels
```

Render visual previews:

```bash
yolo-mosaic visualize \
  --images-dir output/images \
  --labels-dir output/labels \
  --output-dir output/visualized
```

## Web UI Usage

```bash
yolo-mosaic web --host 127.0.0.1 --port 7860
```

Open [http://127.0.0.1:7860](http://127.0.0.1:7860). The Gradio interface reports file matching by filename stem, highlights missing or extra annotations, supports class-name files, and exports the raw mosaic, visualized mosaic, annotation text, and ZIP bundle.

Web export files are stored in the system temporary directory under `yolo_mosaic_web_exports` and old export folders are cleaned after a six-hour retention window. Recent exports are intentionally retained so Gradio can finish serving downloads.

## Architecture

See [docs/architecture.md](docs/architecture.md) for the module diagram and design notes.

```text
dataset -> annotations -> geometry -> validation -> mosaic -> services -> CLI/web
```

Core responsibility boundaries:

- `geometry.py`: coordinate conversion, clipping, letterbox transforms, scaling, and offsets.
- `validation.py`: repair, visibility filtering, minimum-size filtering, and statistics.
- `mosaic.py`: deterministic equal-cell 2x2 and 3x3 mosaic composition.
- `services.py`: shared workflows for CLI, web, benchmark, and export paths.
- `web.py`: Gradio controls and user-facing messages only.

## Bounding-Box Transformation Mathematics

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

Letterbox mode uses equal `scale_x` and `scale_y`, plus cell offset and top-left padding. Transformed boxes are then repaired, clipped, and filtered with the same validation configuration used for source annotations.

## Testing And CI

Run the same checks used by CI:

```bash
python -m ruff check .
python -m mypy src
python -m pytest --cov=src/yolo_mosaic --cov-report=term-missing
pre-commit run --all-files
```

GitHub Actions prints installed tool versions, then runs Ruff, mypy, and pytest with coverage on Python 3.11 and 3.12.

## Benchmarking

```bash
yolo-mosaic benchmark --output-dir examples/benchmark --overwrite
```

The benchmark prints environment information and measured timings for synthetic generation, mosaic generation, and visualization. This repository does not include invented benchmark claims.

## Docker Usage

```bash
docker build -t yolo-mosaic-toolkit .
docker run --rm yolo-mosaic-toolkit --help
docker run --rm -p 7860:7860 yolo-mosaic-toolkit web --host 0.0.0.0 --port 7860
```

## Example Outputs

Reproducible examples are documented in [docs/examples.md](docs/examples.md).

- Synthetic dataset: `examples/synthetic_dataset/`
- 2x2 mosaics: `examples/outputs/2x2/`
- 3x3 mosaics: `examples/outputs/3x3/`
- README assets: `docs/assets/`

## Limitations

- The implemented mosaic mode is deterministic equal-cell placement, not randomized YOLO-style mosaic center cropping.
- COCO and Pascal VOC import/export are not included.
- Gradio's built-in file upload dropzone text follows the viewer's browser locale; project-defined labels, help text, and portfolio screenshots are maintained in English.
- Web exports use temporary files with a retention cleanup window; very long-lived download links can expire after cleanup.
- Parallel workers are reserved for future scaling work.

## Roadmap

- Randomized mosaic center with crop-aware visibility filtering.
- COCO and Pascal VOC adapters.
- Dataset statistics dashboard.
- Parallel processing for large datasets.
- Property-based tests for geometry invariants.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT. See [LICENSE](LICENSE).
