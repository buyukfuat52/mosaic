"""Run a measured synthetic benchmark from the repository root."""

from __future__ import annotations

import json
from pathlib import Path

from yolo_mosaic.services import run_benchmark


def main() -> None:
    """Execute the default benchmark and print measured results."""

    payload = run_benchmark(Path("examples/benchmark"), overwrite=True)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
