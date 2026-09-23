"""Select a second EPA AQS network using development-period metadata only.

The selection deliberately excludes Salt Lake City and never reads a 2025 file.
It favors a single-county network so the source query and spatial unit mirror the
original experiment, and it fixes seven stations to preserve the original model
shape and training budget.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aqriskformer.utils import read_json, sha256_file, utc_now, write_json

POLLUTANTS = ("PM2.5", "NO2", "O3", "CO", "SO2")
CORE = ("PM2.5", "NO2", "O3")
DEFAULT_PRESENCE = ROOT / "data_external/epa_aqs/selection_audit/site_pollutant_presence.csv"
DEFAULT_COVERAGE = (
    ROOT / "data_external/epa_aqs/selection_audit/eligible_site_pollutant_coverage.csv"
)
DEFAULT_SUMMARY = ROOT / "data_external/epa_aqs/selection_audit/summary.json"
DEFAULT_OUTPUT = ROOT / "journal_protocol/external_replication_selection.json"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def present(row: dict[str, str], pollutant: str) -> bool:
    return bool(str(row.get(pollutant, "")).strip())


def county_key(site_id: str) -> str:
    state, county, _ = site_id.split("-")
    return f"{state}-{county}"


def network_record(name: str, rows: list[dict[str, str]]) -> dict[str, object]:
    pm25_sites = sum(present(row, "PM2.5") for row in rows)
    relevant_pm25_plus_two = sum(
        present(row, "PM2.5")
        and sum(present(row, pollutant) for pollutant in POLLUTANTS) >= 3
        for row in rows
    )
    core_sites = sum(all(present(row, pollutant) for pollutant in CORE) for row in rows)
    network_pollutants = sum(
        any(present(row, pollutant) for row in rows) for pollutant in POLLUTANTS
    )
    score = (
        5 * core_sites
        + 3 * relevant_pm25_plus_two
        + 2 * pm25_sites
        + network_pollutants
    )
    return {
        "cbsa": name,
        "county_keys": sorted({county_key(row["site_id"]) for row in rows}),
        "eligible_sites": len(rows),
        "pm25_sites": pm25_sites,
        "relevant_pm25_plus_two_sites": relevant_pm25_plus_two,
        "core_pm25_no2_o3_sites": core_sites,
        "network_pollutant_count": network_pollutants,
        "selection_score": score,
    }


def station_record(row: dict[str, str]) -> dict[str, object]:
    available = [pollutant for pollutant in POLLUTANTS if present(row, pollutant)]
    coverages = [float(row[pollutant]) for pollutant in available]
    return {
        "site_id": row["site_id"],
        "available_pollutants": available,
        "relevant_pollutant_count": len(available),
        "core_pm25_no2_o3": all(present(row, pollutant) for pollutant in CORE),
        "pm25_and_o3": present(row, "PM2.5") and present(row, "O3"),
        "mean_minimum_annual_observation_percent": sum(coverages) / len(coverages),
        "minimum_annual_observation_percent_by_pollutant": {
            pollutant: float(row[pollutant]) for pollutant in available
        },
    }


def select_network(
    presence_rows: list[dict[str, str]], station_count: int
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in presence_rows:
        grouped[row["CBSA Name"]].append(row)
    candidates = [
        network_record(name, rows)
        for name, rows in grouped.items()
        if name != "Salt Lake City, UT"
    ]
    candidates = [
        record
        for record in candidates
        if len(record["county_keys"]) == 1
        and int(record["pm25_sites"]) >= station_count
        and int(record["network_pollutant_count"]) == len(POLLUTANTS)
    ]
    candidates.sort(
        key=lambda item: (
            -int(item["selection_score"]),
            -int(item["core_pm25_no2_o3_sites"]),
            -int(item["relevant_pm25_plus_two_sites"]),
            -int(item["pm25_sites"]),
            str(item["cbsa"]),
        )
    )
    if not candidates:
        raise RuntimeError("No external network satisfies the locked eligibility rule")
    selected = candidates[0]
    station_candidates = [
        station_record(row)
        for row in grouped[str(selected["cbsa"])]
        if present(row, "PM2.5")
    ]
    station_candidates.sort(
        key=lambda item: (
            -int(item["relevant_pollutant_count"]),
            -int(bool(item["core_pm25_no2_o3"])),
            -int(bool(item["pm25_and_o3"])),
            -float(item["mean_minimum_annual_observation_percent"]),
            str(item["site_id"]),
        )
    )
    return selected, station_candidates[:station_count], candidates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--presence", type=Path, default=DEFAULT_PRESENCE)
    parser.add_argument("--coverage", type=Path, default=DEFAULT_COVERAGE)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--station-count", type=int, default=7)
    args = parser.parse_args()
    if args.station_count < 4:
        parser.error("station-count must be at least four for the locked 3-NN graph")
    summary = read_json(args.summary)
    if summary.get("holdout_year_inspected") is not False:
        raise RuntimeError("Selection audit does not attest that the holdout was unseen")
    selected, stations, candidates = select_network(
        read_rows(args.presence), args.station_count
    )
    coverage_rows = read_rows(args.coverage)
    metadata: dict[str, dict[str, str]] = {}
    for row in coverage_rows:
        if row["site_id"] in {item["site_id"] for item in stations}:
            metadata.setdefault(row["site_id"], row)
    if set(metadata) != {item["site_id"] for item in stations}:
        raise RuntimeError("Selected station metadata is incomplete")
    state, county = str(selected["county_keys"][0]).split("-")
    record = {
        "status": "EXTERNAL_NETWORK_SELECTED_FROM_2021_2024_ONLY",
        "created_utc": utc_now(),
        "scientific_role": (
            "Pre-specified geographic replication of the frozen Salt Lake candidate; "
            "not a second model-selection exercise."
        ),
        "selection_data": {
            "years": summary["selection_years"],
            "holdout_year_inspected": False,
            "minimum_observation_percent_each_year": summary[
                "minimum_observation_percent_each_year"
            ],
            "presence_table": str(args.presence.resolve().relative_to(ROOT)),
            "presence_sha256": sha256_file(args.presence),
            "coverage_table": str(args.coverage.resolve().relative_to(ROOT)),
            "coverage_sha256": sha256_file(args.coverage),
            "audit_summary": str(args.summary.resolve().relative_to(ROOT)),
            "audit_summary_sha256": sha256_file(args.summary),
        },
        "network_eligibility": {
            "exclude_development_network": "Salt Lake City, UT",
            "single_county_cbsa": True,
            "minimum_pm25_sites": args.station_count,
            "all_five_replication_pollutants_present_somewhere_in_network": True,
            "pollutants": list(POLLUTANTS),
        },
        "network_ranking": {
            "score": (
                "5*core_PM2.5_NO2_O3_sites + 3*PM2.5_plus_any_two_relevant_sites + "
                "2*PM2.5_sites + network_relevant_pollutant_count"
            ),
            "tie_breakers": [
                "core_PM2.5_NO2_O3_sites descending",
                "PM2.5_plus_any_two_relevant_sites descending",
                "PM2.5_sites descending",
                "CBSA name ascending",
            ],
            "eligible_candidate_count": len(candidates),
            "top_candidates": candidates[:10],
        },
        "selected_network": {
            **selected,
            "state_fips": state,
            "county_fips": county,
            "civil_time_zone": "America/Los_Angeles",
            "local_standard_utc_offset_hours": -8,
        },
        "station_selection": {
            "station_count": args.station_count,
            "rule": (
                "Require PM2.5, then sort by relevant pollutant count, complete "
                "PM2.5-NO2-O3 core, PM2.5-plus-O3 availability, mean of the minimum "
                "annual coverage percentages, and site ID. All descending except site ID."
            ),
            "selected_station_ids": [item["site_id"] for item in stations],
            "evidence": [
                {
                    **item,
                    "local_site_name": metadata[str(item["site_id"])]["Local Site Name"],
                    "state_name": metadata[str(item["site_id"])]["State Name"],
                    "county_name": metadata[str(item["site_id"])]["County Name"],
                    "latitude": float(metadata[str(item["site_id"])]["Latitude"]),
                    "longitude": float(metadata[str(item["site_id"])]["Longitude"]),
                }
                for item in stations
            ],
        },
        "replication_invariants": {
            "candidate": "adaptive_graph_impute_tcn",
            "candidate_selected_on_external_data": False,
            "architecture_hyperparameters_calibration_and_endpoints": (
                "Copied unchanged from the frozen Salt Lake protocol before external "
                "development measurements are parsed."
            ),
            "seeds": [42, 123, 2026, 3407, 7777],
            "development_years": [2021, 2022, 2023, 2024],
            "sealed_test_year": 2025,
        },
        "source_protocol_sha256": {
            "journal_protocol/comparison_protocol.json": sha256_file(
                ROOT / "journal_protocol/comparison_protocol.json"
            ),
            "journal_protocol/candidate_selection_freeze.json": sha256_file(
                ROOT / "journal_protocol/candidate_selection_freeze.json"
            ),
            "journal_protocol/final_holdout_protocol.json": sha256_file(
                ROOT / "journal_protocol/final_holdout_protocol.json"
            ),
        },
        "holdout": {
            "year": 2025,
            "downloaded": False,
            "content_parsed": False,
            "metrics_computed": False,
        },
        "next_gate": (
            "Download and validate only 2021-2024 Clark County hourly data, then lock "
            "the development preprocessing protocol before any 2025 download."
        ),
    }
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite selection lock: {args.output}")
    write_json(args.output, record)
    print(f"External network selection locked: {args.output.resolve()}")
    print(f"Selected network: {selected['cbsa']} ({state}-{county})")
    print("Selected stations: " + ", ".join(item["site_id"] for item in stations))
    print(f"SHA-256: {sha256_file(args.output)}")


if __name__ == "__main__":
    main()
