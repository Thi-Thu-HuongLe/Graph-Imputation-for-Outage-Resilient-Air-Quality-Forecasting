"""Development-only preparation for the locked EPA AQS Salt Lake network."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data import (
    PreparedAirQuality,
    _calendar_features,
    _causal_fill,
    _fit_robust_scaler,
    _mase_scale,
    _normalize_static,
    _risk_thresholds,
    _split_index_bounds,
    _time_since_observation,
    _timestamp_mask,
)
from .utils import read_json, sha256_file


@dataclass
class EPADevelopmentNative:
    timestamps_utc: pd.DatetimeIndex
    stations: list[str]
    pollutants: list[str]
    values: np.ndarray
    coordinates: np.ndarray
    merge_report: dict[str, Any]


def _utc_naive(value: str | pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp
    return timestamp.tz_convert("UTC").tz_localize(None)


def _station_id(row: dict[str, Any]) -> str:
    return f"{row['state_code']}-{row['county_code']}-{row['site_number']}"


def _validate_scope(
    parent: dict[str, Any], preprocessing: dict[str, Any], manifest: dict[str, Any]
) -> None:
    dataset, scope = parent["dataset"], preprocessing["scope"]
    if manifest.get("stage") != "development_api" or manifest.get("holdout_data_included"):
        raise ValueError("Refusing a manifest that is not explicitly development-only")
    if scope["stations"] != dataset["station_ids"]:
        raise ValueError("Preprocessing stations differ from the locked station selection")
    if scope["pollutants"] != dataset["pollutants"]:
        raise ValueError("Preprocessing pollutants differ from the locked pollutant order")
    allowed_years = {int(year) for year in scope["allowed_local_years"]}
    forbidden_years = {int(year) for year in scope["forbidden_local_years"]}
    entries = manifest.get("files", [])
    actual = {(entry["pollutant"], int(entry["year"])) for entry in entries}
    expected = {(pollutant, year) for pollutant in scope["pollutants"] for year in allowed_years}
    if len(entries) != len(expected) or actual != expected:
        raise ValueError("Manifest is not the exact pollutant/year development grid")
    if any(int(entry["year"]) in forbidden_years for entry in entries):
        raise ValueError("Manifest contains a forbidden holdout year")


def load_epa_development_native(
    parent_protocol_path: str | Path,
    preprocessing_protocol_path: str | Path,
    raw_root: str | Path,
) -> EPADevelopmentNative:
    """Load and merge only files explicitly enumerated by the development manifest."""
    parent_path = Path(parent_protocol_path)
    preprocessing_path = Path(preprocessing_protocol_path)
    raw_path = Path(raw_root)
    parent = read_json(parent_path)
    preprocessing = read_json(preprocessing_path)
    manifest_path = raw_path / "download_manifest.json"
    manifest = read_json(manifest_path)
    if sha256_file(manifest_path) != parent["dataset"]["development_download_manifest_sha256"]:
        raise ValueError("Development manifest hash differs from the parent protocol")
    _validate_scope(parent, preprocessing, manifest)

    scope = preprocessing["scope"]
    time_axis = preprocessing["time_axis"]
    stations = list(scope["stations"])
    pollutants = list(scope["pollutants"])
    station_lookup = {station: index for index, station in enumerate(stations)}
    parameter_codes = dict(scope["parameter_codes"])
    expected_units = preprocessing["measurement_policy"]["units"]
    timestamps = pd.date_range(
        _utc_naive(time_axis["development_start_utc"]),
        _utc_naive(time_axis["development_end_utc"]),
        freq="h",
    )
    if len(timestamps) != int(time_axis["expected_hours"]):
        raise ValueError("The locked UTC time axis has an unexpected length")
    values = np.full((len(timestamps), len(stations), len(pollutants)), np.nan, dtype=np.float32)
    coordinate_rows: dict[str, list[tuple[float, float]]] = {station: [] for station in stations}
    report: dict[str, Any] = {
        pollutant: {
            "finite_rows": 0,
            "merged_hourly_values": 0,
            "duplicate_groups": 0,
            "duplicate_extra_rows": 0,
            "maximum_multiplicity": 0,
            "maximum_duplicate_range": 0.0,
        }
        for pollutant in pollutants
    }

    entries = sorted(
        manifest["files"],
        key=lambda entry: (int(entry["year"]), pollutants.index(entry["pollutant"])),
    )
    for entry in entries:
        path = raw_path / entry["file"]
        if path.parent.resolve() != raw_path.resolve():
            raise ValueError(f"Manifest path escapes development directory: {entry['file']}")
        if sha256_file(path) != entry["sha256"]:
            raise ValueError(f"Development source hash mismatch: {path}")
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
            if str(row["parameter_code"]) != parameter_codes[pollutant]:
                raise ValueError(f"Unexpected parameter code in {path}")
            if str(row["sample_duration_code"]) != "1":
                raise ValueError(f"Unexpected sample duration in {path}")
            if str(row["units_of_measure"]) != expected_units[pollutant]:
                raise ValueError(f"Unexpected units in {path}")
            coordinate_rows[_station_id(row)].append(
                (float(row["latitude"]), float(row["longitude"]))
            )

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
            raise ValueError(f"AQS timestamp lies outside locked development axis: {bad}")
        station_positions = aggregate["station"].map(station_lookup).to_numpy(dtype=int)
        existing = values[positions, station_positions, pollutant_index]
        if np.isfinite(existing).any():
            raise ValueError(f"Overlapping local-year files for {pollutant}")
        values[positions, station_positions, pollutant_index] = aggregate["median"].to_numpy(
            dtype=np.float32
        )

        counts = aggregate["count"].to_numpy(dtype=int)
        ranges = (aggregate["max"] - aggregate["min"]).to_numpy(dtype=float)
        section = report[pollutant]
        section["finite_rows"] += len(frame)
        section["merged_hourly_values"] += len(aggregate)
        section["duplicate_groups"] += int((counts > 1).sum())
        section["duplicate_extra_rows"] += int(np.maximum(counts - 1, 0).sum())
        section["maximum_multiplicity"] = max(
            int(section["maximum_multiplicity"]), int(counts.max()) if len(counts) else 0
        )
        section["maximum_duplicate_range"] = max(
            float(section["maximum_duplicate_range"]), float(ranges.max()) if len(ranges) else 0.0
        )

    coordinates = np.empty((len(stations), 2), dtype=np.float32)
    for station, station_index in station_lookup.items():
        samples = np.asarray(coordinate_rows[station], dtype=float)
        if not len(samples) or not np.isfinite(samples).all():
            raise ValueError(f"Missing or nonfinite station coordinates: {station}")
        coordinates[station_index] = np.median(samples, axis=0)

    return EPADevelopmentNative(
        timestamps_utc=timestamps,
        stations=stations,
        pollutants=pollutants,
        values=values,
        coordinates=coordinates,
        merge_report=report,
    )


def haversine_distances(coordinates: np.ndarray) -> np.ndarray:
    """Return pairwise great-circle distance in kilometers for latitude/longitude rows."""
    radians = np.deg2rad(np.asarray(coordinates, dtype=float))
    latitude, longitude = radians[:, 0], radians[:, 1]
    d_latitude = latitude[:, None] - latitude[None, :]
    d_longitude = longitude[:, None] - longitude[None, :]
    a = (
        np.sin(d_latitude / 2) ** 2
        + np.cos(latitude[:, None]) * np.cos(latitude[None, :]) * np.sin(d_longitude / 2) ** 2
    )
    return (2 * 6371.0088 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))).astype(np.float32)


def coordinate_knn_graph(
    coordinates: np.ndarray, neighbors: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    distances = haversine_distances(coordinates)
    stations = len(distances)
    if not 0 < neighbors < stations:
        raise ValueError("neighbors must be between one and station_count - 1")
    directed = np.zeros((stations, stations), dtype=bool)
    for station in range(stations):
        order = np.argsort(distances[station], kind="stable")
        chosen = order[order != station][:neighbors]
        directed[station, chosen] = True
    edges = directed | directed.T
    nonzero = distances[np.triu_indices(stations, k=1)]
    bandwidth = float(np.median(nonzero[nonzero > 0]))
    if not math.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError("Station coordinates do not define a nonzero graph bandwidth")
    graph = np.where(edges, np.exp(-((distances / bandwidth) ** 2)), 0.0)
    np.fill_diagonal(graph, 0.0)
    return graph.astype(np.float32), distances


def prepare_epa_development(
    parent_protocol_path: str | Path,
    preprocessing_protocol_path: str | Path,
    raw_root: str | Path,
) -> tuple[PreparedAirQuality, dict[str, Any], pd.DataFrame]:
    preprocessing_path = Path(preprocessing_protocol_path)
    preprocessing = read_json(preprocessing_path)
    native = load_epa_development_native(
        parent_protocol_path, preprocessing_protocol_path, raw_root
    )
    bounds = preprocessing["split_bounds_utc"]
    train_bounds = [_utc_naive(value) for value in bounds["train"]]
    validation_bounds = [_utc_naive(value) for value in bounds["validation"]]
    train_mask = _timestamp_mask(native.timestamps_utc, train_bounds)
    observed = np.isfinite(native.values)
    # Entire partial-pollutant cells are expected (for example, no SO2 monitor at
    # some selected sites). The shared scaler deliberately falls back to the
    # pollutant-wide training statistics for those cells.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="All-NaN slice encountered", category=RuntimeWarning
        )
        center, scale = _fit_robust_scaler(native.values, observed, train_mask)
    fill_limit = int(preprocessing["missingness"]["input_causal_forward_fill_limit_hours"])
    filled = _causal_fill(native.values, center, limit=fill_limit)
    scaled = (filled - center[None]) / scale[None]
    local_standard = native.timestamps_utc - pd.to_timedelta(7, unit="h")
    thresholds = _risk_thresholds(native.values, observed, train_mask)
    mase = _mase_scale(
        native.values,
        observed,
        train_mask,
        season=int(preprocessing["training_only_transforms"]["mase_season_hours"]),
    )
    graph, distances = coordinate_knn_graph(
        native.coordinates, neighbors=int(preprocessing["station_graph"]["neighbors"])
    )
    split_bounds = {
        "train": _split_index_bounds(native.timestamps_utc, train_bounds),
        "validation": _split_index_bounds(native.timestamps_utc, validation_bounds),
    }
    prepared = PreparedAirQuality(
        name="epa_aqs_salt_lake_development",
        timestamps_ns=native.timestamps_utc.to_numpy(dtype="datetime64[ns]"),
        stations=native.stations,
        pollutants=native.pollutants,
        feature_names=native.pollutants,
        meteorology=[],
        values=scaled.astype(np.float32),
        observed_mask=observed,
        time_gaps=_time_since_observation(observed),
        calendar=_calendar_features(local_standard),
        station_static=_normalize_static(native.coordinates),
        native_pollutants=native.values,
        target_mask=observed,
        center=center,
        scale=scale,
        risk_thresholds=thresholds,
        mase_scale24=mase,
        train_correlation_graph=graph,
        split_bounds=split_bounds,
    )

    rows = []
    for station_index, station in enumerate(native.stations):
        for pollutant_index, pollutant in enumerate(native.pollutants):
            item = {
                "station": station,
                "pollutant": pollutant,
                "latitude": float(native.coordinates[station_index, 0]),
                "longitude": float(native.coordinates[station_index, 1]),
            }
            for split, (start, end) in split_bounds.items():
                count = end - start + 1
                observed_count = int(
                    observed[start : end + 1, station_index, pollutant_index].sum()
                )
                item[f"{split}_hours"] = count
                item[f"{split}_observed"] = observed_count
                item[f"{split}_coverage"] = observed_count / count
            rows.append(item)
    availability = pd.DataFrame(rows)
    off_diagonal_edges = int(np.triu(graph > 0, k=1).sum())
    report = {
        "status": "PASS",
        "scope": "EPA AQS 2021-2024 development only; sealed 2025 content not accessed",
        "parent_protocol_sha256": sha256_file(parent_protocol_path),
        "preprocessing_protocol_sha256": sha256_file(preprocessing_path),
        "raw_manifest_sha256": sha256_file(Path(raw_root) / "download_manifest.json"),
        "shape": list(native.values.shape),
        "timestamps_utc": {
            "start": str(native.timestamps_utc[0]),
            "end": str(native.timestamps_utc[-1]),
            "hours": len(native.timestamps_utc),
        },
        "stations": native.stations,
        "pollutants": native.pollutants,
        "coordinates": native.coordinates,
        "split_bounds": split_bounds,
        "split_timestamps_utc": {
            split: [str(native.timestamps_utc[start]), str(native.timestamps_utc[end])]
            for split, (start, end) in split_bounds.items()
        },
        "embargo_hours": int(bounds["embargo_hours"]),
        "observed_fraction_by_pollutant": {
            pollutant: float(observed[:, :, index].mean())
            for index, pollutant in enumerate(native.pollutants)
        },
        "merge_report": native.merge_report,
        "center": center,
        "scale": scale,
        "risk_thresholds": thresholds,
        "mase_scale24": mase,
        "graph": {
            "kind": "coordinate_symmetric_3nn_gaussian",
            "undirected_edges": off_diagonal_edges,
            "adjacency": graph,
            "distances_km": distances,
        },
        "native_values_sha256": hashlib.sha256(native.values.tobytes()).hexdigest(),
        "observed_mask_sha256": hashlib.sha256(observed.tobytes()).hexdigest(),
    }
    return prepared, report, availability
