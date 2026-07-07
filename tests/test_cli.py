from pathlib import Path

from typer.testing import CliRunner

from yolo_mosaic.cli import app


def test_cli_synthetic_generate_visualize_workflow(tmp_path: Path) -> None:
    runner = CliRunner()
    dataset = tmp_path / "dataset"
    result = runner.invoke(
        app,
        ["synthetic", "--output-dir", str(dataset), "--num-images", "8", "--seed", "42"],
    )
    assert result.exit_code == 0, result.output

    output_images = tmp_path / "outputs" / "images"
    output_labels = tmp_path / "outputs" / "labels"
    result = runner.invoke(
        app,
        [
            "generate",
            "--images-dir",
            str(dataset / "images"),
            "--labels-dir",
            str(dataset / "labels"),
            "--output-images-dir",
            str(output_images),
            "--output-labels-dir",
            str(output_labels),
            "--grid",
            "2",
            "--width",
            "200",
            "--height",
            "200",
            "--count",
            "1",
            "--seed",
            "42",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (output_images / "mosaic_0000.jpg").exists()
    assert (output_labels / "mosaic_0000.txt").exists()

    result = runner.invoke(
        app,
        [
            "visualize",
            "--images-dir",
            str(output_images),
            "--labels-dir",
            str(output_labels),
            "--output-dir",
            str(tmp_path / "visualized"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "visualized" / "mosaic_0000.jpg").exists()


def test_cli_validate_command(tmp_path: Path) -> None:
    runner = CliRunner()
    dataset = tmp_path / "dataset"
    result = runner.invoke(app, ["synthetic", "--output-dir", str(dataset), "--num-images", "4"])
    assert result.exit_code == 0, result.output
    result = runner.invoke(
        app,
        [
            "validate",
            "--images-dir",
            str(dataset / "images"),
            "--labels-dir",
            str(dataset / "labels"),
            "--output-labels-dir",
            str(tmp_path / "validated"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "validated" / "synthetic_001.txt").exists()
