"""Freeze the Las Vegas development protocol before any 2025 download or access."""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aqriskformer.utils import read_json, sha256_file, utc_now, write_json

DEFAULT_SELECTION = ROOT / "journal_protocol/external_replication_selection.json"
DEFAULT_RAW = ROOT / "data_external/epa_aqs_las_vegas/raw_development_api"
DEFAULT_PARENT = ROOT / "journal_protocol/external_unseen_holdout_protocol.json"
DEFAULT_PREPROCESSING = (
    ROOT / "journal_protocol/external_development_preprocessing_protocol.json"
)
DEFAULT_COMPARISON = ROOT / "journal_protocol/external_comparison_protocol.json"

POLLUTANTS = ["PM2.5", "NO2", "O3", "CO", "SO2"]
PARAMETER_CODES = {
    "PM2.5": "88101",
    "NO2": "42602",
    "O3": "44201",
    "CO": "42101",
    "SO2": "42401",
}
UNITS = {
    "PM2.5": "Micrograms/cubic meter (LC)",
    "NO2": "Parts per billion",
    "O3": "Parts per million",
    "CO": "Parts per million",
    "SO2": "Parts per billion",
}


def rooted(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT)).replace("\\", "/")


def require_absent(paths: list[Path]) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite frozen protocol(s): " + ", ".join(existing))


def validate_manifest(manifest: dict[str, object], selection: dict[str, object]) -> None:
    network = selection["selected_network"]
    if manifest.get("stage") != "development_api" or manifest.get("holdout_data_included"):
        raise ValueError("Expected a development-only AQS manifest")
    if manifest.get("state_fips") != network["state_fips"] or manifest.get(
        "county_fips"
    ) != network["county_fips"]:
        raise ValueError("Development manifest does not match the selected county")
    expected = {(pollutant, year) for pollutant in POLLUTANTS for year in range(2021, 2025)}
    actual = {
        (str(item["pollutant"]), int(item["year"])) for item in manifest.get("files", [])
    }
    if len(manifest.get("files", [])) != len(expected) or actual != expected:
        raise ValueError("Development manifest is not the exact 5-pollutant x 4-year grid")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--parent-output", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--preprocessing-output", type=Path, default=DEFAULT_PREPROCESSING)
    parser.add_argument("--comparison-output", type=Path, default=DEFAULT_COMPARISON)
    args = parser.parse_args()
    require_absent(
        [args.parent_output, args.preprocessing_output, args.comparison_output]
    )
    selection = read_json(args.selection)
    if selection.get("status") != "EXTERNAL_NETWORK_SELECTED_FROM_2021_2024_ONLY":
        raise RuntimeError("External network selection is not locked")
    if selection["holdout"] != {
        "year": 2025,
        "downloaded": False,
        "content_parsed": False,
        "metrics_computed": False,
    }:
        raise RuntimeError("Selection record does not preserve the unopened holdout gate")
    manifest_path = args.raw_dir / "download_manifest.json"
    manifest = read_json(manifest_path)
    validate_manifest(manifest, selection)
    stations = list(selection["station_selection"]["selected_station_ids"])
    network = selection["selected_network"]

    parent = {
        "protocol_status": "EXTERNAL_DATASET_AND_SPLITS_LOCKED_HOLDOUT_NOT_DOWNLOADED",
        "created_utc": utc_now(),
        "purpose": (
            "Geographic replication of the already frozen Salt Lake candidate on a "
            "second EPA AQS network. No model selection is permitted on this network."
        ),
        "selection_lock": rooted(args.selection),
        "selection_lock_sha256": sha256_file(args.selection),
        "dataset": {
            "name": "US EPA AQS Las Vegas-Clark County hourly sensor network",
            "source_url": "https://aqs.epa.gov/aqsweb/airdata/download_files.html",
            "api_endpoint": "https://aqs.epa.gov/data/api/sampleData/byCounty",
            "license_or_terms": (
                "US EPA public monitoring data; preserve source attribution and "
                "retrieval metadata"
            ),
            "download_or_snapshot_date": str(manifest["created_utc"])[:10],
            "development_download_manifest": rooted(manifest_path),
            "development_download_manifest_sha256": sha256_file(manifest_path),
            "development_validation_summary": rooted(
                args.raw_dir / "validation_summary.json"
            ),
            "development_validation_status_at_freeze": "NOT_YET_RUN",
            "holdout_download_manifest": "NOT_DOWNLOADED",
            "state_fips": network["state_fips"],
            "county_fips": network["county_fips"],
            "civil_time_zone": network["civil_time_zone"],
            "local_standard_utc_offset_hours": network[
                "local_standard_utc_offset_hours"
            ],
            "time_zone": (
                "America/Los_Angeles civil clock audited against AQS local fields; "
                "fixed UTC-08:00 local standard time used for the canonical calendar"
            ),
            "units": UNITS,
            "sample_duration_code": "1",
            "sample_duration": "1 HOUR",
            "station_count": len(stations),
            "station_ids": stations,
            "station_selection_rule": selection["station_selection"]["rule"],
            "station_coordinates_available": True,
            "pollutants": POLLUTANTS,
            "common_core_pollutants": ["PM2.5", "NO2", "O3"],
            "partial_pollutants": ["CO", "SO2"],
            "target_policy": (
                "Score every prespecified station-pollutant cell with an observed target; "
                "also report pollutant and availability strata."
            ),
            "natural_missingness_retained": True,
        },
        "split_policy": {
            "type": "strict_chronological",
            "train": "2021-01-01 through 2023-12-31 fixed local standard time",
            "validation": "2024-01-01 through 2024-12-31 fixed local standard time",
            "test": "2025-01-01 through 2025-12-31; not downloaded until final freeze",
            "embargo_hours_between_splits": 168,
            "test_access_rule": (
                "Do not download or parse 2025 until external development runs, exact "
                "candidate/checkpoint paths, calibration, inference, and reporting code "
                "are frozen."
            ),
        },
        "forecast": {
            "lookback_hours": 168,
            "maximum_horizon_hours": 48,
            "reported_horizons_hours": [1, 6, 12, 24, 48],
            "target_pollutants": POLLUTANTS,
        },
        "candidate": "adaptive_graph_impute_tcn",
        "candidate_selected_on_this_network": False,
        "required_comparators": [
            "persistence",
            "seasonal_naive_24h",
            "local_tcn",
            "local_capacity_residual",
            "impute_then_local_tcn",
            "dcrnn",
            "graph_wavenet",
        ],
        "seeds": [42, 123, 2026, 3407, 7777],
        "primary_conditions": [
            "target_station_trailing_6h_outage",
            "target_station_trailing_24h_outage",
        ],
        "primary_endpoints": ["mase", "q95_brier"],
        "calibration": {
            "method": (
                "per-pollutant multiplicative Gaussian sigma factor fitted on clean "
                "2024 validation predictions"
            ),
            "duration_specific_calibration": False,
            "test_label_fitting": False,
        },
        "reporting_rule": (
            "Report all prespecified external-network results regardless of direction; "
            "do not revise the model from external 2025 outcomes."
        ),
    }

    preprocessing = {
        "status": "EXTERNAL_PREPROCESSING_LOCKED_BEFORE_2025_DOWNLOAD",
        "parent_protocol": rooted(args.parent_output),
        "purpose": "Deterministic preparation of Clark County 2021-2024 development data.",
        "prepared_dataset_name": "epa_aqs_las_vegas_development",
        "scope": {
            "manifest": rooted(manifest_path),
            "allowed_local_years": [2021, 2022, 2023, 2024],
            "forbidden_local_years": [2025],
            "stations": stations,
            "pollutants": POLLUTANTS,
            "parameter_codes": PARAMETER_CODES,
        },
        "time_axis": {
            "canonical": "continuous hourly UTC",
            "stored_representation": (
                "timezone-naive datetime64[ns] whose values denote UTC"
            ),
            "calendar_clock": "AQS local standard time, fixed UTC-08:00",
            "local_standard_utc_offset_hours": -8,
            "development_start_utc": "2021-01-01T08:00:00Z",
            "development_end_utc": "2025-01-01T07:00:00Z",
            "expected_hours": 35064,
            "audit_requirement": (
                "development audit must report zero fixed-local-standard clock mismatches"
            ),
        },
        "measurement_policy": {
            "accepted_sample_duration_code": "1",
            "accepted_measurements": (
                "finite numeric sample_measurement values returned by AQS"
            ),
            "qualifiers": (
                "retain finite measurements regardless of qualifier and report qualifier "
                "prevalence; nonfinite or absent measurements remain missing"
            ),
            "negative_measurements": (
                "retain as reported instrument values; do not clamp or reinterpret as missing"
            ),
            "concurrent_monitor_merge": (
                "median across finite POC/method readings sharing station, pollutant, and UTC hour"
            ),
            "units": UNITS,
        },
        "split_bounds_utc": {
            "train": ["2021-01-01T08:00:00Z", "2024-01-01T07:00:00Z"],
            "embargo_not_scored": [
                "2024-01-01T08:00:00Z",
                "2024-01-08T07:00:00Z",
            ],
            "validation": ["2024-01-08T08:00:00Z", "2025-01-01T07:00:00Z"],
            "embargo_hours": 168,
            "operational_lookback_rule": (
                "Validation inputs may use observations before the first validation target, "
                "but no embargo timestamp is a training or validation target."
            ),
        },
        "missingness": {
            "native_mask": "preserve before filling",
            "input_causal_forward_fill_limit_hours": 6,
            "fallback_after_limit": (
                "training station-pollutant median, represented by zero after scaling"
            ),
            "time_since_observation": (
                "recomputed causally after every experimental outage; log1p at model input"
            ),
            "target_rule": "score only finite observed targets",
        },
        "training_only_transforms": {
            "scaler": (
                "station-pollutant median and IQR; pollutant-wide training fallback for "
                "empty or constant cells"
            ),
            "exceedance_quantiles": [0.8, 0.9, 0.95],
            "exceedance_pooling": (
                "each pollutant pooled across all selected stations in the training split"
            ),
            "mase_season_hours": 24,
            "mase_denominator": (
                "station-pollutant mean absolute seasonal difference over jointly observed "
                "training pairs"
            ),
        },
        "station_graph": {
            "source": "AQS WGS84 station coordinates only",
            "distance": "haversine kilometers",
            "neighbors": 3,
            "construction": "symmetric union of directed 3-nearest-neighbor edges",
            "edge_weight": (
                "exp(-(distance / median_nonzero_pairwise_distance)^2)"
            ),
            "diagonal": 0,
            "outcome_values_used": False,
        },
        "development_audit": (
            "data_external/epa_aqs_las_vegas/development_audit/content_audit.json"
        ),
        "prepared_output": (
            "data_external/epa_aqs_las_vegas/prepared_development/"
            "las_vegas_2021_2024.npz"
        ),
        "preprocessing_report": (
            "data_external/epa_aqs_las_vegas/prepared_development/"
            "preprocessing_report.json"
        ),
        "availability_table": (
            "data_external/epa_aqs_las_vegas/prepared_development/availability.csv"
        ),
    }

    comparison = copy.deepcopy(read_json(ROOT / "journal_protocol/comparison_protocol.json"))
    comparison.update(
        status="EXTERNAL_DEVELOPMENT_PROTOCOL_LOCKED_HOLDOUT_NOT_DOWNLOADED",
        name="las_vegas_external_outage_resilience_replication",
        prepared_development=preprocessing["prepared_output"],
        candidate="adaptive_graph_impute_tcn",
        learned_models=[
            "local_tcn",
            "local_capacity_residual",
            "impute_then_local_tcn",
            "dcrnn",
            "graph_wavenet",
        ],
    )
    comparison["replication_lock"] = {
        "network_selection": rooted(args.selection),
        "network_selection_sha256": sha256_file(args.selection),
        "parent_protocol": rooted(args.parent_output),
        "preprocessing_protocol": rooted(args.preprocessing_output),
        "candidate_fixed_from_salt_lake": True,
        "hyperparameters_changed": False,
        "external_holdout_downloaded": False,
        "external_holdout_accessed": False,
        "source_salt_lake_candidate_freeze_sha256": sha256_file(
            ROOT / "journal_protocol/candidate_selection_freeze.json"
        ),
        "source_salt_lake_final_protocol_sha256": sha256_file(
            ROOT / "journal_protocol/final_holdout_protocol.json"
        ),
    }
    comparison["reporting"]["holdout_rule"] = (
        "Do not download or parse Clark County 2025 data until all external development "
        "runs, calibration files, final checkpoint paths, and evaluation code are frozen."
    )

    write_json(args.parent_output, parent)
    write_json(args.preprocessing_output, preprocessing)
    write_json(args.comparison_output, comparison)
    print(f"External parent protocol: {args.parent_output.resolve()}")
    print(f"External preprocessing protocol: {args.preprocessing_output.resolve()}")
    print(f"External comparison protocol: {args.comparison_output.resolve()}")
    print("2025 external holdout remains not downloaded.")


if __name__ == "__main__":
    main()
