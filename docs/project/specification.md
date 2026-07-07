# PROJECT SPECIFICATION — YOLO Mosaic Annotation Toolkit

## Purpose

Create a professional Python toolkit that generates 2x2 and 3x3 mosaic images from YOLO object detection datasets while correctly transforming, validating, repairing, visualizing, and exporting bounding-box annotations.

The project is intended for a public GitHub portfolio and should demonstrate strong software engineering and computer vision fundamentals.

## Default technical decisions

- Python 3.11+
- `src/` package layout
- Typer CLI
- Gradio web UI
- OpenCV and NumPy
- pytest
- Ruff
- mypy
- GitHub Actions
- Docker
- pre-commit
- Deterministic equal-cell grid mosaic
- Letterbox placement
- Padding color `(114, 114, 114)`
- Minimum visible-area ratio `0.25`
- Minimum box size `2x2 px`
- Missing label policy `empty`
- Insufficient input policy `repeat`
- Seeded deterministic behavior

## Core invariants

1. Internal bounding-box representation is pixel XYXY.
2. Geometry calculations use floating-point precision.
3. Normalization occurs only when target image dimensions are known.
4. Valid boxes satisfy `x_max > x_min` and `y_max > y_min`.
5. Final boxes remain within mosaic boundaries.
6. UI and CLI call a shared service layer.
7. Serialization precision is configurable and defaults to 6 decimal places.
8. Same seed, same inputs, and same configuration produce the same result.
9. Image transformation parameters and bounding-box transformation parameters come from the same transform object.

## Primary modules

| Module | Responsibility |
|---|---|
| `models.py` | Typed data models |
| `annotations.py` | YOLO parsing and serialization |
| `geometry.py` | Coordinate conversion and box transforms |
| `validation.py` | Repair, clipping, area, and visibility rules |
| `image_ops.py` | Image loading, resizing, and letterboxing |
| `dataset.py` | Image-label matching and iteration |
| `mosaic.py` | 2x2 and 3x3 composition |
| `visualization.py` | Bounding-box rendering |
| `synthetic.py` | Synthetic dataset generation |
| `reporting.py` | Statistics and manifest output |
| `services.py` | Shared application workflows |
| `cli.py` | Typer commands |
| `web.py` | Gradio interface |

## Required repository support

- `README.md`
- `docs/portfolio/portfolio-guide.md`
- `CONTRIBUTING.md`
- `CHANGELOG.md`
- `LICENSE`
- `Dockerfile`
- `Makefile`
- `.pre-commit-config.yaml`
- `.github/workflows/ci.yml`
- `docs/architecture.md`
- `docs/examples.md`
- benchmark script
- synthetic demo data
- generated demo outputs

## Portfolio goals

The repository should visibly demonstrate:

- Computer vision geometry
- Data validation
- Reproducible augmentation
- CLI design
- Web integration
- Testing discipline
- CI/CD basics
- Packaging
- Containerization
- Documentation
- Performance awareness
