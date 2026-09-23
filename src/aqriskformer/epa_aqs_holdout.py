"""One-time EPA AQS 2025 holdout preparation guarded by a final hash freeze."""

from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data import (
    PreparedAirQuality,
    _calendar_features,
    _causal_fill,
    _time_since_observation,
)
from .outage_data import OutageData
from .utils import read_json, sha256_file


@dataclass
class EPAHoldoutNative:
    timestamps_utc: pd.DatetimeIndex
    stations: list[str]
    pollutants: list[str]
    values: np.ndarray
    merge_report: dict[str, Any]


def _rooted(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _utc_naive(value: str | pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return timestamp if timestamp.tzinfo is None else timestamp.tz_convert("UTC").tz_localize(None)


def _station_id(row: dict[str, Any]) -> str:
    return f"{row['state_code']}-{row['county_code']}-{row['site_number']}"


def verify_execution_freeze(root: str | Path, freeze_path: str | Path) -> dict[str, Any]:
    """Verify every frozen source/input hash before holdout content can be opened."""
    project_root = Path(root).resolve()
    path = Path(freeze_path).resolve()
    freeze = read_json(path)
    if freeze.get("status") != "FINAL_HOLDOUT_EXECUTION_FROZEN_READY_TO_OPEN":
        raise RuntimeError("Final holdout execution is not frozen")
    if freeze.get("holdout_evaluation_authorized") is not True:
        raise RuntimeError("Freeze record does not authorize holdout evaluation")
    if freeze.get("holdout_content_parsed_at_freeze") is not False:
        raise RuntimeError("Freeze record does not attest an unopened holdout")
    for section in ("source_sha256", "input_sha256"):
        for name, expected in freeze[section].items():
            candidate = _rooted(project_root, name)
            if not candidate.is_file() or sha256_file(candidate) != expected:
                raise RuntimeError(f"Post-freeze hash mismatch: {candidate}")
    return freeze


def validate_holdout_manifest(
    manifest: dict[str, Any],
    stations: list[str],
    pollutants: list[str],
    parameter_codes: dict[str, str],
) -> list[dict[str, Any]]:
    """Validate the sealed manifest without discovering additional files."""
    if manifest.get("stage") != "holdout_api" or manifest.get("holdout_data_included") is not True:
        raise ValueError("Manifest is not the designated holdout download")
    entries = list(manifest.get("files", []))
    actual = {(str(item["pollutant"]), int(item["year"])) for item in entries}
    expected = {(pollutant, 2025) for pollutant in pollutants}
    if len(entries) != len(expected) or actual != expected:
        raise ValueError("Holdout manifest is not the exact pollutant/year grid")
    for entry in entries:
        pollutant = str(entry["pollutant"])
        if str(entry["parameter_code"]) != parameter_codes[pollutant]:
            raise ValueError(f"Parameter-code mismatch for {pollutant}")
        if str(entry["sample_duration_code"]) != "1":
            raise ValueError(f"Non-hourly manifest entry for {pollutant}")
        if not entry.get("sha256"):
            raise ValueError(f"Missing source hash for {pollutant}")
    if not stations:
        raise ValueError("The frozen station list is empty")
    return sorted(entries, key=lambda item: pollutants.index(str(item["pollutant"])))


def load_epa_holdout_native(
    protocol_path: str | Path,
    preprocessing_path: str | Path,
    raw_root: str | Path,
) -> EPAHoldoutNative:
    """Open only manifest-enumerated 2025 sources after the execution freeze gate."""
    protocol = read_json(protocol_path)
    preprocessing = read_json(preprocessing_path)
    raw = Path(raw_root).resolve()
    manifest_path = raw / "download_manifest.json"
    if sha256_file(manifest_path) != protocol["data"]["sealed_manifest_sha256"]:
        raise ValueError("Sealed holdout manifest hash mismatch")
    manifest = read_json(manifest_path)
    scope = preprocessing["scope"]
    stations = list(scope["stations"])
    pollutants = list(scope["pollutants"])
    entries = validate_holdout_manifest(
        manifest, stations, pollutants, dict(scope["parameter_codes"])
    )
    timestamps = pd.date_range(
        _utc_naive(protocol["data"]["holdout_start_utc"]),
        _utc_naive(protocol["data"]["holdout_end_utc"]),
        freq="h",
    )
    if len(timestamps) != int(protocol["data"]["expected_holdout_hours"]):
        raise ValueError("Locked holdout time axis has an unexpected length")

    station_lookup = {station: index for index, station in enumerate(stations)}
    values = np.full((len(timestamps), len(stations), len(pollutants)), np.nan, np.float32)
    expected_units = preprocessing["measurement_policy"]["units"]
    report: dict[str, Any] = {}
    for entry in entries:
        path = (raw / str(entry["file"])).resolve()
        if path.parent != raw:
            raise ValueError(f"Manifest path escapes sealed directory: {entry['file']}")
        if sha256_file(path) != entry["sha256"]:
            raise ValueError(f"Sealed source hash mismatch: {path}")
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        header = payload.get("Header", [{}])[0]
        rows = payload.get("Data", [])
        if header.get("status") != "Success" or int(header.get("rows", -1)) != len(rows):
            raise ValueError(f"Invalid AQS response: {path}")
        pollutant = str(entry["pollutant"])
        pollutant_index = pollutants.index(pollutant)
        selected = [row for row in rows if _station_id(row) in station_lookup]
        for row in selected:
            if str(row["parameter_code"]) != scope["parameter_codes"][pollutant]:
                raise ValueError(f"Unexpected parameter code in {path}")
            if str(row["sample_duration_code"]) != "1":
                raise ValueError(f"Unexpected sample duration in {path}")
            if str(row["units_of_measure"]) != expected_units[pollutant]:
                raise ValueError(f"Unexpected units in {path}")
        frame = pd.DataFrame(
            {
                "station": [_station_id(row) for row in selected],
                "timestamp": pd.to_datetime(
                    [f"{row['date_gmt']} {row['time_gmt']}" for row in selected], utc=True
                ).tz_localize(None),
                "measurement": pd.to_numeric(
                    [row.get("sample_measurement") for row in selected], errors="coerce"
                ),
            }
        )
        finite = np.isfinite(frame["measurement"].to_numpy(dtype=float))
        frame = frame.loc[finite]
        grouped = frame.groupby(["station", "timestamp"], sort=False)["measurement"]
        aggregate = grouped.agg(["count", "median", "min", "max"]).reset_index()
        positions = timestamps.get_indexer(pd.DatetimeIndex(aggregate["timestamp"]))
        if (positions < 0).any():
            bad = aggregate.loc[positions < 0, "timestamp"].iloc[0]
            raise ValueError(f"AQS timestamp lies outside locked holdout axis: {bad}")
        station_positions = aggregate["station"].map(station_lookup).to_numpy(dtype=int)
        if np.isfinite(values[positions, station_positions, pollutant_index]).any():
            raise ValueError(f"Overlapping holdout sources for {pollutant}")
        values[positions, station_positions, pollutant_index] = aggregate["median"].to_numpy(
            np.float32
        )
        counts = aggregate["count"].to_numpy(dtype=int)
        ranges = (aggregate["max"] - aggregate["min"]).to_numpy(dtype=float)
        report[pollutant] = {
            "api_rows": len(rows),
            "selected_rows": len(selected),
            "finite_rows": len(frame),
            "merged_hourly_values": len(aggregate),
            "duplicate_groups": int((counts > 1).sum()),
            "duplicate_extra_rows": int(np.maximum(counts - 1, 0).sum()),
            "maximum_multiplicity": int(counts.max()) if len(counts) else 0,
            "maximum_duplicate_range": float(ranges.max()) if len(ranges) else 0.0,
        }
        del payload, rows, selected, frame, aggregate
    return EPAHoldoutNative(timestamps, stations, pollutants, values, report)


def prepare_epa_holdout(
    protocol_path: str | Path,
    preprocessing_path: str | Path,
    raw_root: str | Path,
    development_prepared_path: str | Path,
) -> tuple[PreparedAirQuality, dict[str, Any], pd.DataFrame]:
    """Apply only frozen development transforms to the untouched 2025 period."""
    protocol = read_json(protocol_path)
    development_path = Path(development_prepared_path)
    if sha256_file(development_path) != protocol["data"]["development_prepared_sha256"]:
        raise ValueError("Development prepared artifact hash mismatch")
    development = PreparedAirQuality.load(development_path)
    native = load_epa_holdout_native(protocol_path, preprocessing_path, raw_root)
    if native.stations != development.stations or native.pollutants != development.pollutants:
        raise ValueError("Holdout station/pollutant order differs from development")
    context_hours = int(protocol["data"]["context_hours_from_development"])
    embargo_hours = int(protocol["data"]["embargo_hours_not_scored"])
    if context_hours < int(protocol["forecast"]["lookback"]):
        raise ValueError("Frozen context is shorter than the model lookback")
    context_timestamps = pd.DatetimeIndex(development.timestamps[-context_hours:])
    if context_timestamps[-1] + pd.Timedelta(hours=1) != native.timestamps_utc[0]:
        raise ValueError("Development context and holdout are not hourly-contiguous")
    timestamps = context_timestamps.append(native.timestamps_utc)
    native_values = np.concatenate(
        [development.native_pollutants[-context_hours:], native.values], axis=0
    ).astype(np.float32)
    observed = np.isfinite(native_values)
    fill_limit = int(protocol["forecast"]["fill_limit"])
    filled = _causal_fill(native_values, development.center, limit=fill_limit)
    scaled = (filled - development.center[None]) / development.scale[None]
    local_standard = timestamps - pd.to_timedelta(7, unit="h")
    test_start = context_hours + embargo_hours
    split_bounds = {"test": (test_start, len(timestamps) - 1)}
    prepared = PreparedAirQuality(
        name="epa_aqs_salt_lake_2025_holdout",
        timestamps_ns=timestamps.to_numpy(dtype="datetime64[ns]"),
        stations=development.stations,
        pollutants=development.pollutants,
        feature_names=development.feature_names,
        meteorology=development.meteorology,
        values=scaled.astype(np.float32),
        observed_mask=observed,
        time_gaps=_time_since_observation(observed),
        calendar=_calendar_features(local_standard),
        station_static=development.station_static,
        native_pollutants=native_values,
        target_mask=observed,
        center=development.center,
        scale=development.scale,
        risk_thresholds=development.risk_thresholds,
        mase_scale24=development.mase_scale24,
        train_correlation_graph=development.train_correlation_graph,
        split_bounds=split_bounds,
    )
    first_scored = timestamps[test_start]
    if first_scored != _utc_naive(protocol["data"]["first_scored_target_utc"]):
        raise ValueError("First scored timestamp differs from the frozen protocol")

    rows: list[dict[str, Any]] = []
    holdout_observed = np.isfinite(native.values)
    for station_index, station in enumerate(native.stations):
        for pollutant_index, pollutant in enumerate(native.pollutants):
            count = len(native.timestamps_utc)
            observed_count = int(holdout_observed[:, station_index, pollutant_index].sum())
            rows.append(
                {
                    "station": station,
                    "pollutant": pollutant,
                    "holdout_hours": count,
                    "holdout_observed": observed_count,
                    "holdout_coverage": observed_count / count,
                }
            )
    availability = pd.DataFrame(rows)
    report = {
        "status": "PASS_HOLDOUT_PREPARED_AFTER_FINAL_FREEZE",
        "holdout_content_accessed": True,
        "test_metrics_computed": False,
        "protocol_sha256": sha256_file(protocol_path),
        "preprocessing_protocol_sha256": sha256_file(preprocessing_path),
        "raw_manifest_sha256": sha256_file(Path(raw_root) / "download_manifest.json"),
        "development_prepared_sha256": sha256_file(development_path),
        "shape_including_context": list(prepared.values.shape),
        "context_hours": context_hours,
        "embargo_hours_not_scored": embargo_hours,
        "test_split_bounds": split_bounds["test"],
        "test_target_timestamps_utc": [str(timestamps[test_start]), str(timestamps[-1])],
        "stations": native.stations,
        "pollutants": native.pollutants,
        "observed_fraction_by_pollutant": {
            pollutant: float(holdout_observed[:, :, index].mean())
            for index, pollutant in enumerate(native.pollutants)
        },
        "merge_report": native.merge_report,
        "native_holdout_values_sha256": hashlib.sha256(native.values.tobytes()).hexdigest(),
        "native_holdout_mask_sha256": hashlib.sha256(holdout_observed.tobytes()).hexdigest(),
    }
    return prepared, report, availability


def load_holdout_outage_data(path: str | Path) -> OutageData:
    """Load a prepared test-only artifact without weakening the development loader."""
    prepared = PreparedAirQuality.load(path)
    if set(prepared.split_bounds) != {"test"}:
        raise ValueError("Holdout artifact must contain exactly one test split")
    raw = np.where(prepared.observed_mask, prepared.values, 0.0).astype(np.float32)
    data = OutageData(
        raw=raw,
        observed=prepared.observed_mask,
        calendar=prepared.calendar,
        center=prepared.center,
        scale=prepared.scale,
        thresholds=prepared.risk_thresholds,
        mase=prepared.mase_scale24,
        timestamps=prepared.timestamps_ns,
        stations=prepared.stations,
        pollutants=prepared.pollutants,
        graph=prepared.train_correlation_graph,
        split_bounds=prepared.split_bounds,
    )
    if data.raw.shape != data.observed.shape or data.raw.shape[0] != len(data.timestamps):
        raise ValueError("Prepared holdout tensor dimensions are inconsistent")
    if data.center.shape != data.scale.shape or not (data.scale > 0).all():
        raise ValueError("Frozen scaling statistics are invalid")
    if data.graph.shape != (len(data.stations), len(data.stations)):
        raise ValueError("Frozen station graph has invalid dimensions")
    start, end = data.split_bounds["test"]
    if not 0 <= start <= end < len(data.timestamps):
        raise ValueError("Holdout split bounds are invalid")
    return data
