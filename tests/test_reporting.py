import json
from pathlib import Path

from yolo_mosaic.models import ProcessingStats
from yolo_mosaic.reporting import write_json_report, write_manifest


def test_reporting_writes_json_and_manifest(tmp_path: Path) -> None:
    stats = ProcessingStats(valid_boxes=2)
    report = tmp_path / "reports" / "stats.json"
    write_json_report(report, stats, {"name": "demo"})
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["stats"]["valid_boxes"] == 2
    assert payload["name"] == "demo"

    manifest = tmp_path / "manifest.csv"
    write_manifest(manifest, [{"image": "a.jpg", "label": "a.txt"}])
    assert "image,label" in manifest.read_text(encoding="utf-8")


def test_empty_manifest_is_not_written(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    write_manifest(manifest, [])
    assert not manifest.exists()
