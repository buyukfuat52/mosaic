# Changelog

## 0.1.0

- Added typed data models for YOLO boxes, pixel boxes, mosaic configuration, validation configuration, transforms, and processing statistics.
- Added YOLO annotation parsing and serialization.
- Added centralized geometry conversion, clipping, letterbox, and transform functions.
- Added validation and repair for reversed, out-of-bounds, non-finite, zero-area, too-small, and low-visibility boxes.
- Added deterministic 2x2 and 3x3 letterboxed mosaic generation.
- Added dataset loading, synthetic data generation, visualization, service workflows, Typer CLI, Gradio web UI, benchmark support, tests, CI, Docker, and documentation.
- Fixed validation configuration propagation so transformed mosaic boxes use the same custom thresholds as source annotations.
- Added regression coverage for post-transform minimum width, minimum height, visible-ratio, and default validation behavior.
- Aligned pre-commit Mypy dependencies with the CI NumPy constraint and included `pre-commit` in the development extra.
- Improved the Gradio UI with upload matching summaries, warnings, class-name support, demo generation, friendlier errors, progress labels, and web export cleanup.
- Reorganized project-generation documentation under `docs/project`, `docs/portfolio`, and `docs/ai-development`.
- Added real generated portfolio assets and refreshed README and example documentation for the v0.1.0 release.
