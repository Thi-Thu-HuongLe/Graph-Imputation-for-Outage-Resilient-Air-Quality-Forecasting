from __future__ import annotations

from pathlib import Path

import pytest

from aqriskformer.epa_aqs_holdout import (
    validate_holdout_manifest,
    verify_execution_freeze,
)
from aqriskformer.utils import sha256_file, write_json

STATIONS = ["49-035-2005"]
POLLUTANTS = ["PM2.5", "NO2"]
CODES = {"PM2.5": "88101", "NO2": "42602"}


def manifest() -> dict[str, object]:
    return {
        "stage": "holdout_api",
        "holdout_data_included": True,
        "files": [
            {
                "file": "pm25.gz",
                "pollutant": "PM2.5",
                "year": 2025,
                "parameter_code": "88101",
                "sample_duration_code": "1",
                "sha256": "a",
            },
            {
                "file": "no2.gz",
                "pollutant": "NO2",
                "year": 2025,
                "parameter_code": "42602",
                "sample_duration_code": "1",
                "sha256": "b",
            },
        ],
    }


def test_holdout_manifest_requires_exact_locked_grid() -> None:
    entries = validate_holdout_manifest(manifest(), STATIONS, POLLUTANTS, CODES)
    assert [entry["pollutant"] for entry in entries] == POLLUTANTS
    incomplete = manifest()
    incomplete["files"] = incomplete["files"][:1]
    with pytest.raises(ValueError, match="exact pollutant/year grid"):
        validate_holdout_manifest(incomplete, STATIONS, POLLUTANTS, CODES)


def test_holdout_manifest_rejects_parameter_change() -> None:
    changed = manifest()
    changed["files"][0]["parameter_code"] = "99999"
    with pytest.raises(ValueError, match="Parameter-code mismatch"):
        validate_holdout_manifest(changed, STATIONS, POLLUTANTS, CODES)


def test_execution_freeze_detects_post_freeze_change(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    source.write_text("frozen", encoding="utf-8")
    freeze = tmp_path / "freeze.json"
    record = {
        "status": "FINAL_HOLDOUT_EXECUTION_FROZEN_READY_TO_OPEN",
        "holdout_evaluation_authorized": True,
        "holdout_content_parsed_at_freeze": False,
        "source_sha256": {"source.txt": sha256_file(source)},
        "input_sha256": {},
    }
    write_json(freeze, record)
    assert verify_execution_freeze(tmp_path, freeze)["holdout_evaluation_authorized"]
    source.write_text("changed", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Post-freeze hash mismatch"):
        verify_execution_freeze(tmp_path, freeze)
