# Contributing

Thank you for improving the YOLO Mosaic Annotation Toolkit.

## Development Setup

```bash
python -m pip install -e ".[dev]"
```

## Quality Checks

Run the same checks used by CI:

```bash
ruff check .
mypy src
pytest --cov=src/yolo_mosaic --cov-report=term-missing
```

## Engineering Rules

- Keep bounding-box math in `src/yolo_mosaic/geometry.py`.
- Keep validation and repair behavior in `src/yolo_mosaic/validation.py`.
- Add numerical tests for geometry changes.
- Do not overwrite user data by default.
- Use `pathlib.Path` for filesystem paths.
- Update docs when behavior changes.

## Pull Request Guidance

Include a short summary, test results that were actually run, and any behavior or documentation changes. If a command was not run, state that directly.
