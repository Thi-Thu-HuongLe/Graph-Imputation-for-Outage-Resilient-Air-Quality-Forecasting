"""Seal the downloaded Clark County 2025 manifest and write the final protocol."""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aqriskformer.utils import read_json, sha256_file, utc_now, write_json


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT)).replace("\\", "/")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--development-freeze",
        type=Path,
        default=ROOT / "journal_protocol/external_development_runs_freeze.json",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=ROOT / "data_external/epa_aqs_las_vegas/raw_holdout_sealed",
    )
    parser.add_argument(
        "--protocol-output",
        type=Path,
        default=ROOT / "journal_protocol/external_final_holdout_protocol.json",
    )
    parser.add_argument(
        "--seal-output",
        type=Path,
        default=ROOT / "journal_protocol/external_holdout_seal_status.json",
    )
    args = parser.parse_args()
    if args.protocol_output.exists() or args.seal_output.exists():
        raise FileExistsError("Refusing to overwrite the external final protocol or seal")
    development = read_json(args.development_freeze)
    if development.get("status") != "EXTERNAL_DEVELOPMENT_RUNS_FROZEN_2025_NOT_DOWNLOADED":
        raise RuntimeError("External development runs are not frozen")
    if development.get("holdout_download_authorized") is not True:
        raise RuntimeError("External holdout download was not authorized")
    manifest_path = args.raw_dir / "download_manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("stage") != "holdout_api" or manifest.get("holdout_data_included") is not True:
        raise ValueError("Manifest is not a sealed holdout download")
    if manifest.get("state_fips") != "32" or manifest.get("county_fips") != "003":
        raise ValueError("Holdout manifest is not Clark County, Nevada")
    entries = list(manifest.get("files", []))
    expected = {(name, 2025) for name in ("PM2.5", "NO2", "O3", "CO", "SO2")}
    actual = {(str(item["pollutant"]), int(item["year"])) for item in entries}
    if len(entries) != len(expected) or actual != expected:
        raise ValueError("Holdout manifest is not the exact 2025 pollutant grid")
    for item in entries:
        path = args.raw_dir / str(item["file"])
        if not path.is_file() or sha256_file(path) != item["sha256"]:
            raise RuntimeError(f"Sealed source hash mismatch: {path}")

    source_final = read_json(ROOT / "journal_protocol/final_holdout_protocol.json")
    final = copy.deepcopy(source_final)
    final.update(
        status="EXTERNAL_FINAL_HOLDOUT_PROTOCOL_LOCKED_CONTENT_NOT_PARSED",
        name="las_vegas_2025_external_geographic_replication",
        candidate_selection_freeze=rel(args.development_freeze),
        candidate_selection_freeze_sha256=sha256_file(args.development_freeze),
        checkpoint_roots=development["checkpoint_roots"],
        output="experiment_protocol/results_external/holdout_2025",
        development_protocol="journal_protocol/external_comparison_protocol.json",
    )
    final["data"] = {
        "development_prepared": development["prepared_development"],
        "development_prepared_sha256": development["prepared_development_sha256"],
        "sealed_raw_root": rel(args.raw_dir),
        "sealed_manifest": rel(manifest_path),
        "sealed_manifest_sha256": sha256_file(manifest_path),
        "prepared_holdout": (
            "data_external/epa_aqs_las_vegas/prepared_holdout/las_vegas_2025.npz"
        ),
        "prepared_dataset_name": "epa_aqs_las_vegas_2025_holdout",
        "preparation_report": (
            "data_external/epa_aqs_las_vegas/prepared_holdout/preparation_report.json"
        ),
        "availability_table": (
            "data_external/epa_aqs_las_vegas/prepared_holdout/availability.csv"
        ),
        "holdout_start_utc": "2025-01-01T08:00:00Z",
        "holdout_end_utc": "2026-01-01T07:00:00Z",
        "expected_holdout_hours": 8760,
        "context_hours_from_development": 168,
        "embargo_hours_not_scored": 168,
        "first_scored_target_utc": "2025-01-08T08:00:00Z",
        "calendar_clock": "fixed UTC-08:00 local standard time",
        "local_standard_utc_offset_hours": -8,
    }
    final["access_rule"] = (
        "The external 2025 gzip/JSON files may be opened only after the external final "
        "execution freeze exists, authorizes evaluation, and every recorded hash verifies."
    )
    final["rerun_rule"] = (
        "Report all external results regardless of direction. Never revise the candidate, "
        "calibration, endpoint family, or protocol using Clark County 2025 labels."
    )
    write_json(args.protocol_output, final)

    seal = {
        "status": "EXTERNAL_DOWNLOADED_AND_SEALED_NOT_PARSED",
        "created_utc": utc_now(),
        "holdout_manifest": rel(manifest_path),
        "holdout_manifest_sha256": sha256_file(manifest_path),
        "final_protocol": rel(args.protocol_output),
        "final_protocol_sha256": sha256_file(args.protocol_output),
        "gzip_or_json_content_opened": False,
        "concentration_values_inspected": False,
        "test_metrics_computed": False,
        "file_sha256": {str(item["file"]): item["sha256"] for item in entries},
    }
    write_json(args.seal_output, seal)
    print(f"External final protocol: {args.protocol_output.resolve()}")
    print(f"External holdout seal: {args.seal_output.resolve()}")
    print("The 2025 concentration content remains unopened.")


if __name__ == "__main__":
    main()
