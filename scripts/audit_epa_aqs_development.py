"""Audit only the unsealed EPA AQS 2021--2024 development measurements.

This script deliberately reads files listed by the development manifest only.  It
does not discover, open, or summarize anything in the sealed 2025 directory.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "journal_protocol/unseen_holdout_protocol.json"
DEFAULT_RAW = ROOT / "data_external/epa_aqs/raw_development_api"
DEFAULT_OUTPUT = ROOT / "data_external/epa_aqs/development_audit/content_audit.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def qualifier_key(value: object) -> str:
    if value is None or value == "" or value == []:
        return "<NONE>"
    if isinstance(value, list):
        return "|".join(sorted(str(item) for item in value))
    return str(value)


def audit(protocol_path: Path, raw_root: Path) -> dict[str, object]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    manifest_path = raw_root / "download_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset = protocol["dataset"]
    civil_time_zone = str(dataset.get("civil_time_zone", "America/Denver"))
    local_standard_offset = float(dataset.get("local_standard_utc_offset_hours", -7.0))
    if not -14.0 <= local_standard_offset <= 14.0:
        raise ValueError(f"Invalid fixed UTC offset: {local_standard_offset}")
    stations = list(dataset["station_ids"])
    station_set = set(stations)
    pollutants = list(dataset["pollutants"])
    units_expected = dict(dataset["units"])

    if manifest.get("stage") != "development_api" or manifest.get("holdout_data_included"):
        raise ValueError("Refusing a manifest that is not development-only")
    if sha256_file(manifest_path) != dataset["development_download_manifest_sha256"]:
        raise ValueError("Development manifest hash differs from the locked protocol")

    expected_pairs = {(pollutant, year) for pollutant in pollutants for year in range(2021, 2025)}
    actual_pairs = {(entry["pollutant"], int(entry["year"])) for entry in manifest["files"]}
    if actual_pairs != expected_pairs or len(manifest["files"]) != len(expected_pairs):
        raise ValueError("Development manifest must contain exactly 5 pollutants x 4 years")

    qualifier_counts: Counter[tuple[str, str]] = Counter()
    units: defaultdict[str, set[str]] = defaultdict(set)
    methods: defaultdict[tuple[str, str], set[tuple[str, str, str]]] = defaultdict(set)
    coordinates: defaultdict[str, list[tuple[float, float]]] = defaultdict(list)
    summary_by_pollutant: dict[str, dict[str, object]] = {
        pollutant: {
            "api_rows": 0,
            "selected_station_rows": 0,
            "finite_measurements": 0,
            "qualified_finite_measurements": 0,
            "negative_finite_measurements": 0,
            "hourly_station_groups": 0,
            "duplicate_hourly_station_groups": 0,
            "duplicate_extra_rows": 0,
            "maximum_group_multiplicity": 0,
            "civil_time_utc_clock_mismatches": 0,
            "local_standard_time_utc_clock_mismatches": 0,
            "minimum_measurement": None,
            "maximum_measurement": None,
            "station_finite_rows": Counter(),
            "station_hourly_groups": Counter(),
        }
        for pollutant in pollutants
    }
    per_file: list[dict[str, object]] = []

    for entry in sorted(
        manifest["files"], key=lambda item: (item["year"], pollutants.index(item["pollutant"]))
    ):
        if int(entry["year"]) >= 2025:
            raise ValueError("Refusing to parse a holdout-year file")
        path = raw_root / entry["file"]
        if sha256_file(path) != entry["sha256"]:
            raise ValueError(f"Hash mismatch: {path}")
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        header = payload.get("Header", [{}])[0]
        rows = payload.get("Data", [])
        if header.get("status") != "Success" or int(header.get("rows", -1)) != len(rows):
            raise ValueError(f"Invalid AQS response header: {path}")

        pollutant = entry["pollutant"]
        selected = [
            row
            for row in rows
            if f"{row['state_code']}-{row['county_code']}-{row['site_number']}" in station_set
        ]
        frame = pd.DataFrame(
            {
                "station": [
                    f"{row['state_code']}-{row['county_code']}-{row['site_number']}"
                    for row in selected
                ],
                "timestamp_utc": pd.to_datetime(
                    [f"{row['date_gmt']} {row['time_gmt']}" for row in selected], utc=True
                ),
                "timestamp_local_reported": pd.to_datetime(
                    [f"{row['date_local']} {row['time_local']}" for row in selected]
                ),
                "measurement": pd.to_numeric(
                    [row.get("sample_measurement") for row in selected], errors="coerce"
                ),
                "qualifier": [qualifier_key(row.get("qualifier")) for row in selected],
            }
        )
        finite = np.isfinite(frame["measurement"].to_numpy(dtype=float))
        finite_frame = frame.loc[finite].copy()
        group_sizes = finite_frame.groupby(["station", "timestamp_utc"], sort=False).size()
        derived_local = frame["timestamp_utc"].dt.tz_convert(civil_time_zone).dt.tz_localize(None)
        civil_clock_mismatches = int((derived_local != frame["timestamp_local_reported"]).sum())
        derived_local_standard = frame["timestamp_utc"].dt.tz_localize(None) + pd.to_timedelta(
            local_standard_offset, unit="h"
        )
        standard_clock_mismatches = int(
            (derived_local_standard != frame["timestamp_local_reported"]).sum()
        )

        for row, keep in zip(selected, finite, strict=True):
            station = f"{row['state_code']}-{row['county_code']}-{row['site_number']}"
            units[pollutant].add(str(row["units_of_measure"]))
            methods[(pollutant, station)].add(
                (str(row.get("poc")), str(row.get("method_code")), str(row.get("method_type")))
            )
            coordinates[station].append((float(row["latitude"]), float(row["longitude"])))
            qualifier_counts[(pollutant, qualifier_key(row.get("qualifier")))] += 1
            if keep:
                summary_by_pollutant[pollutant]["station_finite_rows"][station] += 1
        for station, _ in group_sizes.index:
            summary_by_pollutant[pollutant]["station_hourly_groups"][station] += 1

        values = finite_frame["measurement"].to_numpy(dtype=float)
        qualified = finite_frame["qualifier"].ne("<NONE>").to_numpy()
        section = summary_by_pollutant[pollutant]
        section["api_rows"] += len(rows)
        section["selected_station_rows"] += len(selected)
        section["finite_measurements"] += int(finite.sum())
        section["qualified_finite_measurements"] += int(qualified.sum())
        section["negative_finite_measurements"] += int((values < 0).sum())
        section["hourly_station_groups"] += len(group_sizes)
        section["duplicate_hourly_station_groups"] += int((group_sizes > 1).sum())
        section["duplicate_extra_rows"] += int((group_sizes - 1).clip(lower=0).sum())
        section["maximum_group_multiplicity"] = max(
            int(section["maximum_group_multiplicity"]),
            int(group_sizes.max()) if len(group_sizes) else 0,
        )
        section["civil_time_utc_clock_mismatches"] += civil_clock_mismatches
        section["local_standard_time_utc_clock_mismatches"] += standard_clock_mismatches
        if len(values):
            minimum, maximum = float(values.min()), float(values.max())
            current_min, current_max = (
                section["minimum_measurement"],
                section["maximum_measurement"],
            )
            section["minimum_measurement"] = (
                minimum if current_min is None else min(current_min, minimum)
            )
            section["maximum_measurement"] = (
                maximum if current_max is None else max(current_max, maximum)
            )
        per_file.append(
            {
                "file": entry["file"],
                "sha256": entry["sha256"],
                "pollutant": pollutant,
                "year": int(entry["year"]),
                "api_rows": len(rows),
                "selected_station_rows": len(selected),
                "finite_measurements": int(finite.sum()),
                "hourly_station_groups": len(group_sizes),
                "duplicate_hourly_station_groups": int((group_sizes > 1).sum()),
                "civil_time_utc_clock_mismatches": civil_clock_mismatches,
                "local_standard_time_utc_clock_mismatches": standard_clock_mismatches,
            }
        )

    for pollutant, section in summary_by_pollutant.items():
        finite_count = int(section["finite_measurements"])
        section["qualified_fraction_of_finite"] = (
            float(section["qualified_finite_measurements"] / finite_count) if finite_count else None
        )
        section["negative_fraction_of_finite"] = (
            float(section["negative_finite_measurements"] / finite_count) if finite_count else None
        )
        section["station_finite_rows"] = dict(sorted(section["station_finite_rows"].items()))
        section["station_hourly_groups"] = dict(sorted(section["station_hourly_groups"].items()))

    coordinate_summary = {}
    for station in stations:
        values = np.asarray(coordinates[station], dtype=float)
        coordinate_summary[station] = {
            "latitude_median": float(np.median(values[:, 0])),
            "longitude_median": float(np.median(values[:, 1])),
            "latitude_range": [float(values[:, 0].min()), float(values[:, 0].max())],
            "longitude_range": [float(values[:, 1].min()), float(values[:, 1].max())],
            "records": len(values),
        }

    qualifier_report = {
        pollutant: {
            qualifier: count
            for (item_pollutant, qualifier), count in sorted(qualifier_counts.items())
            if item_pollutant == pollutant
        }
        for pollutant in pollutants
    }
    method_report = {
        f"{pollutant}|{station}": [
            {"poc": poc, "method_code": code, "method_type": method_type}
            for poc, code, method_type in sorted(values)
        ]
        for (pollutant, station), values in sorted(methods.items())
    }
    return {
        "status": "PASS",
        "scope": "EPA AQS development data only; 2025 holdout not accessed",
        "protocol": str(protocol_path.relative_to(ROOT)),
        "protocol_sha256": sha256_file(protocol_path),
        "manifest": str(manifest_path.relative_to(ROOT)),
        "manifest_sha256": sha256_file(manifest_path),
        "stations": stations,
        "pollutants": pollutants,
        "civil_time_zone": civil_time_zone,
        "local_standard_utc_offset_hours": local_standard_offset,
        "units": {pollutant: sorted(values) for pollutant, values in units.items()},
        "unit_checks": {
            pollutant: sorted(units[pollutant]) == [units_expected[pollutant]]
            for pollutant in pollutants
        },
        "by_pollutant": summary_by_pollutant,
        "qualifiers": qualifier_report,
        "monitor_methods": method_report,
        "station_coordinates": coordinate_summary,
        "files": per_file,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = audit(args.protocol.resolve(), args.raw_dir.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "output": str(args.output.resolve()),
                "pollutants": {
                    pollutant: {
                        "finite": section["finite_measurements"],
                        "hourly_groups": section["hourly_station_groups"],
                        "duplicate_groups": section["duplicate_hourly_station_groups"],
                        "qualified_fraction": section["qualified_fraction_of_finite"],
                        "negative_fraction": section["negative_fraction_of_finite"],
                    }
                    for pollutant, section in report["by_pollutant"].items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
