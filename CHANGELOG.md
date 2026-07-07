# Changelog

## 0.1.0

- Added typed data models for YOLO boxes, pixel boxes, mosaic configuration, validation configuration, transforms, and processing statistics.
- Added YOLO annotation parsing and serialization.
- Added centralized geometry conversion, clipping, letterbox, and transform functions.
- Added validation and repair for reversed, out-of-bounds, non-finite, zero-area, too-small, and low-visibility boxes.
- Added deterministic 2x2 and 3x3 letterboxed mosaic generation.
- Added dataset loading, synthetic data generation, visualization, service workflows, Typer CLI, Gradio web UI, benchmark support, tests, CI, Docker, and documentation.
