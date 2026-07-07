# EXECUTION PROMPT

Build the complete **YOLO Mosaic Annotation Toolkit** in this workspace according to:

- `docs/ai-development/system-prompt.md`
- `docs/project/specification.md`
- `docs/project/acceptance-criteria.md`

This is a portfolio project intended to strengthen a computer vision and software engineering resume.

Execution order:

1. Inspect the repository.
2. If the repository is empty, create the recommended `src/` architecture.
3. Present a concise implementation plan and repository tree.
4. Implement data models, annotation parsing, geometry, and validation first.
5. Add and run focused tests for the core modules.
6. Implement deterministic 2x2 and 3x3 letterbox mosaic generation.
7. Add and run numerical mosaic coordinate tests.
8. Implement shared application services.
9. Implement the Typer CLI.
10. Implement the Gradio web interface.
11. Implement synthetic dataset generation.
12. Add benchmark support.
13. Add README, architecture documentation, examples, CI, Docker, pre-commit, Makefile, contributing guide, changelog, and portfolio guide.
14. Generate real example outputs from the synthetic dataset.
15. Run all quality checks.
16. Fix failures before reporting completion.

Required verification commands:

```bash
ruff check .
mypy src
pytest --cov=src/yolo_mosaic --cov-report=term-missing
```

Also run at least one end-to-end workflow:

```bash
yolo-mosaic synthetic --output-dir examples/synthetic_dataset --num-images 30 --seed 42
yolo-mosaic generate --images-dir examples/synthetic_dataset/images --labels-dir examples/synthetic_dataset/labels --output-images-dir examples/outputs/2x2/images --output-labels-dir examples/outputs/2x2/labels --grid 2 --count 2 --seed 42
yolo-mosaic visualize --images-dir examples/outputs/2x2/images --labels-dir examples/outputs/2x2/labels --output-dir examples/outputs/2x2/visualized
```

Rules:

- Do not leave placeholders.
- Do not duplicate geometry logic in CLI or web code.
- Do not ask for approval after every file.
- Add regression tests for bugs found during implementation.
- Use deterministic seeded randomness.
- Preserve cross-platform compatibility.
- Do not claim a test, benchmark, coverage result, or CI status unless it was actually measured.
- Treat `docs/project/acceptance-criteria.md` as the final definition of done.

In the final report, include:

- implemented features
- repository structure
- test status
- coverage result
- lint result
- type-check result
- end-to-end demo result
- generated output locations
- known limitations
- recommended future improvements
- portfolio highlights
