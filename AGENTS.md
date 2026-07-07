# AGENTS.md

## Repository objective

Build and maintain a production-quality YOLO mosaic annotation toolkit suitable for a public computer vision engineering portfolio.

## Working rules

1. Inspect existing code and tests before editing.
2. Keep geometry, validation, I/O, CLI, and UI responsibilities separate.
3. Never implement bounding-box mathematics directly in CLI or web modules.
4. Any geometry change must include or update numerical tests.
5. Add regression tests for bug fixes whenever practical.
6. Do not overwrite user data by default.
7. Use local or injected random-number generators.
8. Handle `NaN`, infinity, zero-area boxes, reversed coordinates, and out-of-bounds boxes.
9. Do not leave `TODO`, `pass`, fake output, broken imports, or placeholder implementations.
10. Do not report a check as successful unless it was executed.
11. Preserve unrelated files in non-empty repositories.
12. Prefer small, cohesive functions.
13. Keep the public API typed.
14. Maintain cross-platform paths with `pathlib`.
15. Update documentation when behavior changes.

## Required checks

```bash
python -m ruff check .
python -m mypy src
python -m pytest --cov=src/yolo_mosaic --cov-report=term-missing
```

## Critical files

- `src/yolo_mosaic/geometry.py`
- `src/yolo_mosaic/validation.py`
- `src/yolo_mosaic/mosaic.py`
- `src/yolo_mosaic/services.py`
- `tests/test_geometry.py`
- `tests/test_validation.py`
- `tests/test_mosaic_2x2.py`
- `tests/test_mosaic_3x3.py`

## Portfolio standards

- Keep README examples reproducible.
- Do not invent benchmark numbers.
- Do not use fake badges.
- Generate screenshots and output images from the real implementation.
- Document design trade-offs and limitations.
- Keep `docs/portfolio/portfolio-guide.md` aligned with implemented features.
