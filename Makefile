.PHONY: install lint type test check synthetic demo benchmark web

install:
	python -m pip install -e ".[dev]"

lint:
	ruff check .

type:
	mypy src

test:
	pytest --cov=src/yolo_mosaic --cov-report=term-missing

check: lint type test

synthetic:
	yolo-mosaic synthetic --output-dir examples/synthetic_dataset --num-images 30 --seed 42

demo:
	yolo-mosaic generate --images-dir examples/synthetic_dataset/images --labels-dir examples/synthetic_dataset/labels --output-images-dir examples/outputs/images --output-labels-dir examples/outputs/labels --grid 2 --count 2 --seed 42
	yolo-mosaic visualize --images-dir examples/outputs/images --labels-dir examples/outputs/labels --output-dir examples/outputs/visualized

benchmark:
	yolo-mosaic benchmark --output-dir examples/benchmark --overwrite

web:
	yolo-mosaic web
