"""Audit natural site-wide core-pollutant co-missingness on validation data only."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aqriskformer.outage_data import (
    OutageWindows,
    load_outage_data,
    natural_comissingness_origins,
)
from aqriskformer.utils import read_json, sha256_file, write_json

PROTOCOL = ROOT / "journal_protocol/comparison_protocol.json"
OUTPUT = ROOT / "data_external/epa_aqs/development_audit/natural_comissingness.json"


def main() -> None:
    protocol = read_json(PROTOCOL)
    prepared_path = ROOT / protocol["prepared_development"]
    data = load_outage_data(prepared_path, development_only=True)
    event_protocol = {**protocol, "validation_stride": 1}
    windows = OutageWindows(data, event_protocol, "validation")
    horizons = list(protocol["reported_horizons"])
    report = {
        "status": "PASS",
        "scope": "2024 validation targets only; sealed 2025 content not accessed",
        "definition": (
            "At the forecast origin, PM2.5, NO2, and O3 are all missing at the target "
            "station for at least the specified trailing duration, at least one other "
            "station has a core-pollutant observation, and the target station resumes a "
            "core reading one hour later. Each origin is therefore the event-aligned last "
            "missing hour. This is observable natural co-missingness and is not asserted "
            "to be verified hardware failure."
        ),
        "prepared_sha256": sha256_file(prepared_path),
        "protocol_sha256": sha256_file(PROTOCOL),
        "validation_forecast_origins": len(windows),
        "validation_stride_hours": 1,
        "by_duration": {},
    }
    for hours in (6, 24):
        selected = natural_comissingness_origins(data, windows.origins, hours)
        by_station = {}
        for station_index, station in enumerate(data.stations):
            positions = np.flatnonzero(selected[:, station_index])
            target_counts = {}
            for horizon in horizons:
                target_indices = windows.origins[positions] + horizon
                target_counts[str(horizon)] = int(
                    data.observed[target_indices, station_index, : data.n_pollutants].sum()
                )
            by_station[station] = {
                "forecast_origins": len(positions),
                "origin_timestamps_utc": [
                    str(data.timestamps[origin]) for origin in windows.origins[positions]
                ],
                "observed_targets_by_horizon": target_counts,
            }
        report["by_duration"][str(hours)] = {
            "station_origin_pairs": int(selected.sum()),
            "stations_with_origins": int((selected.sum(axis=0) > 0).sum()),
            "by_station": by_station,
        }
    write_json(OUTPUT, report)
    print(json.dumps(report["by_duration"], indent=2))
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
