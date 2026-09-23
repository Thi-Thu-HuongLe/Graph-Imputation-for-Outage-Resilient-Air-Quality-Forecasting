"""Authorize one holdout run only after code, inputs, and reporting are hash-frozen."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aqriskformer.utils import read_json, sha256_file, utc_now, write_json

PROTOCOL = ROOT / "journal_protocol/final_holdout_protocol.json"
CANDIDATE_FREEZE = ROOT / "journal_protocol/candidate_selection_freeze.json"
OUTPUT = ROOT / "journal_protocol/final_holdout_execution_freeze.json"


def verify_mapping(values: dict[str, str]) -> None:
    for name, expected in values.items():
        path = ROOT / name
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"Pre-freeze artifact mismatch: {path}")


def main() -> None:
    if OUTPUT.exists():
        existing = read_json(OUTPUT)
        if existing.get("status") != "FINAL_HOLDOUT_EXECUTION_FROZEN_READY_TO_OPEN":
            raise RuntimeError(f"Unexpected existing final freeze: {OUTPUT}")
        print(f"Final holdout execution is already frozen: {OUTPUT}")
        print(f"SHA-256: {sha256_file(OUTPUT)}")
        return

    protocol = read_json(PROTOCOL)
    candidate = read_json(CANDIDATE_FREEZE)
    if protocol.get("status") != "FINAL_HOLDOUT_PROTOCOL_LOCKED_CONTENT_NOT_PARSED":
        raise RuntimeError("Final holdout protocol is not locked")
    if sha256_file(CANDIDATE_FREEZE) != protocol["candidate_selection_freeze_sha256"]:
        raise RuntimeError("Candidate-selection freeze differs from the final protocol")
    if candidate.get("status") != "CANDIDATE_SELECTION_FROZEN_HOLDOUT_UNOPENED":
        raise RuntimeError("Candidate selection is not frozen")
    verify_mapping(candidate["source_sha256"])
    verify_mapping(candidate["run_artifact_sha256"])

    holdout_seal = read_json(ROOT / "journal_protocol/holdout_seal_status.json")
    if holdout_seal.get("status") != "DOWNLOADED_AND_SEALED_NOT_PARSED":
        raise RuntimeError("Holdout seal status is not intact")
    if any(
        holdout_seal.get(key) is not False
        for key in (
            "gzip_or_json_content_opened",
            "concentration_values_inspected",
            "test_metrics_computed",
        )
    ):
        raise RuntimeError("Seal record indicates previous holdout access")
    manifest = ROOT / protocol["data"]["sealed_manifest"]
    if sha256_file(manifest) != protocol["data"]["sealed_manifest_sha256"]:
        raise RuntimeError("Sealed holdout manifest hash mismatch")
    prepared = ROOT / protocol["data"]["prepared_holdout"]
    report = ROOT / protocol["data"]["preparation_report"]
    access_log = ROOT / protocol["output"] / "test_access_log.json"
    if any(path.exists() for path in (prepared, report, access_log)):
        raise RuntimeError(
            "Holdout preparation/access artifacts already exist; do not create a new freeze"
        )

    run_source = ROOT / "scripts/run_journal_holdout.py"
    if "fit_multiplicative_sigma_calibration" in run_source.read_text(encoding="utf-8"):
        raise RuntimeError("Holdout runner contains a forbidden test-time calibration fitter")
    source_paths = (
        PROTOCOL,
        CANDIDATE_FREEZE,
        ROOT / "journal_protocol/comparison_protocol.json",
        ROOT / "journal_protocol/development_preprocessing_protocol.json",
        ROOT / "journal_protocol/holdout_seal_status.json",
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
        run_source,
        ROOT / "scripts/analyze_journal_holdout.py",
        Path(__file__).resolve(),
        ROOT / "pyproject.toml",
    )
    input_hashes = {
        **candidate["run_artifact_sha256"],
        str(manifest.relative_to(ROOT)): sha256_file(manifest),
        protocol["data"]["development_prepared"]: sha256_file(
            ROOT / protocol["data"]["development_prepared"]
        ),
    }
    freeze = {
        "status": "FINAL_HOLDOUT_EXECUTION_FROZEN_READY_TO_OPEN",
        "created_utc": utc_now(),
        "protocol": str(PROTOCOL.relative_to(ROOT)),
        "protocol_sha256": sha256_file(PROTOCOL),
        "candidate": protocol["candidate"],
        "comparators": [
            *protocol["learned_comparators"],
            *protocol["deterministic_comparators"],
        ],
        "seeds": protocol["seeds"],
        "source_sha256": {str(path.relative_to(ROOT)): sha256_file(path) for path in source_paths},
        "input_sha256": input_hashes,
        "calibration": {
            "source": "frozen 2024 clean-validation calibration.npz per model/seed",
            "test_time_fitting": False,
        },
        "statistics": protocol["statistics"],
        "reporting_code_frozen": True,
        "holdout_content_parsed_at_freeze": False,
        "holdout_metrics_computed_at_freeze": False,
        "holdout_evaluation_authorized": True,
        "authorization_scope": (
            "One prespecified preparation/evaluation/reporting campaign. Results must be "
            "reported regardless of direction; no test-driven model revision is allowed."
        ),
    }
    write_json(OUTPUT, freeze)
    print(f"Final holdout execution frozen: {OUTPUT}")
    print(f"SHA-256: {sha256_file(OUTPUT)}")
    print("The next command may open the sealed 2025 content exactly as frozen.")


if __name__ == "__main__":
    main()
