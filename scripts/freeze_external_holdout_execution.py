"""Authorize one external holdout evaluation after all code and inputs are frozen."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aqriskformer.utils import read_json, sha256_file, utc_now, write_json

PROTOCOL = ROOT / "journal_protocol/external_final_holdout_protocol.json"
DEVELOPMENT_FREEZE = ROOT / "journal_protocol/external_development_runs_freeze.json"
SEAL = ROOT / "journal_protocol/external_holdout_seal_status.json"
OUTPUT = ROOT / "journal_protocol/external_final_holdout_execution_freeze.json"


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT)).replace("\\", "/")


def verify_mapping(values: dict[str, str]) -> None:
    for name, expected in values.items():
        path = ROOT / name
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"Pre-freeze artifact mismatch: {path}")


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite external execution freeze: {OUTPUT}")
    protocol = read_json(PROTOCOL)
    development = read_json(DEVELOPMENT_FREEZE)
    seal = read_json(SEAL)
    if protocol.get("status") != "EXTERNAL_FINAL_HOLDOUT_PROTOCOL_LOCKED_CONTENT_NOT_PARSED":
        raise RuntimeError("External final protocol is not locked")
    if development.get("status") != "EXTERNAL_DEVELOPMENT_RUNS_FROZEN_2025_NOT_DOWNLOADED":
        raise RuntimeError("External development freeze is invalid")
    if sha256_file(DEVELOPMENT_FREEZE) != protocol["candidate_selection_freeze_sha256"]:
        raise RuntimeError("External development freeze differs from the final protocol")
    verify_mapping(development["source_sha256"])
    verify_mapping(development["run_artifact_sha256"])
    if seal.get("status") != "EXTERNAL_DOWNLOADED_AND_SEALED_NOT_PARSED":
        raise RuntimeError("External holdout seal is not intact")
    if any(
        seal.get(key) is not False
        for key in (
            "gzip_or_json_content_opened",
            "concentration_values_inspected",
            "test_metrics_computed",
        )
    ):
        raise RuntimeError("External seal indicates prior holdout access")
    manifest = ROOT / protocol["data"]["sealed_manifest"]
    if sha256_file(manifest) != protocol["data"]["sealed_manifest_sha256"]:
        raise RuntimeError("External sealed manifest hash mismatch")
    prepared = ROOT / protocol["data"]["prepared_holdout"]
    report = ROOT / protocol["data"]["preparation_report"]
    access_log = ROOT / protocol["output"] / "test_access_log.json"
    if any(path.exists() for path in (prepared, report, access_log)):
        raise RuntimeError("External holdout access artifacts already exist")

    source_paths = [
        PROTOCOL,
        DEVELOPMENT_FREEZE,
        SEAL,
        ROOT / "journal_protocol/external_replication_selection.json",
        ROOT / "journal_protocol/external_unseen_holdout_protocol.json",
        ROOT / "journal_protocol/external_development_preprocessing_protocol.json",
        ROOT / "journal_protocol/external_comparison_protocol.json",
        ROOT / "src/aqriskformer/data.py",
        ROOT / "src/aqriskformer/epa_aqs_holdout.py",
        ROOT / "src/aqriskformer/outage_data.py",
        ROOT / "src/aqriskformer/outage_evaluation.py",
        ROOT / "src/aqriskformer/outage_models.py",
        ROOT / "src/aqriskformer/development_analysis.py",
        ROOT / "src/aqriskformer/statistics.py",
        ROOT / "src/aqriskformer/utils.py",
        ROOT / "src/aqriskformer/models/baselines.py",
        ROOT / "src/aqriskformer/models/graph_baselines.py",
        ROOT / "scripts/run_journal_development.py",
        ROOT / "scripts/run_journal_holdout.py",
        ROOT / "scripts/analyze_journal_holdout.py",
        ROOT / "scripts/run_external_holdout.py",
        ROOT / "scripts/analyze_external_holdout.py",
        Path(__file__).resolve(),
        ROOT / "pyproject.toml",
    ]
    input_hashes = {
        **development["run_artifact_sha256"],
        rel(manifest): sha256_file(manifest),
        protocol["data"]["development_prepared"]: sha256_file(
            ROOT / protocol["data"]["development_prepared"]
        ),
    }
    record = {
        "status": "FINAL_HOLDOUT_EXECUTION_FROZEN_READY_TO_OPEN",
        "replication": "Las Vegas-Clark County external network",
        "created_utc": utc_now(),
        "protocol": rel(PROTOCOL),
        "protocol_sha256": sha256_file(PROTOCOL),
        "candidate": protocol["candidate"],
        "comparators": [
            *protocol["learned_comparators"],
            *protocol["deterministic_comparators"],
        ],
        "seeds": protocol["seeds"],
        "source_sha256": {rel(path): sha256_file(path) for path in source_paths},
        "input_sha256": input_hashes,
        "calibration": {
            "source": "external clean-2024 validation calibration.npz per model/seed",
            "method_fixed_from_salt_lake": True,
            "test_time_fitting": False,
        },
        "statistics": protocol["statistics"],
        "reporting_code_frozen": True,
        "holdout_content_parsed_at_freeze": False,
        "holdout_metrics_computed_at_freeze": False,
        "holdout_evaluation_authorized": True,
        "authorization_scope": (
            "One prespecified external preparation/evaluation/reporting campaign. Report "
            "all results; no external-test-driven model or calibration revision."
        ),
    }
    write_json(OUTPUT, record)
    print(f"External holdout execution frozen: {OUTPUT.resolve()}")
    print(f"SHA-256: {sha256_file(OUTPUT)}")
    print("The next external holdout command may open the sealed 2025 content once.")


if __name__ == "__main__":
    main()
