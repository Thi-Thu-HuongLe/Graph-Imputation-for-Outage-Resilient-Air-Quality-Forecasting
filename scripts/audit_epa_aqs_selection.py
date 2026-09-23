"""Rank candidate EPA AQS metropolitan sensor networks using 2021--2024 only."""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import pandas as pd

POLLUTANTS = {
    "88101": "PM2.5",
    "81102": "PM10",
    "42602": "NO2",
    "44201": "O3",
    "42101": "CO",
    "42401": "SO2",
}
YEARS = tuple(range(2021, 2025))


def read_annual(zip_path: Path, year: int) -> pd.DataFrame:
    member = f"annual_conc_by_monitor_{year}.csv"
    columns = [
        "State Code",
        "County Code",
        "Site Num",
        "Parameter Code",
        "Sample Duration",
        "Observation Count",
        "Observation Percent",
        "Local Site Name",
        "State Name",
        "County Name",
        "City Name",
        "CBSA Name",
        "Latitude",
        "Longitude",
    ]
    with zipfile.ZipFile(zip_path) as archive, archive.open(member) as handle:
        frame = pd.read_csv(
            handle,
            dtype={
                "State Code": "string",
                "County Code": "string",
                "Site Num": "string",
                "Parameter Code": "string",
            },
            usecols=columns,
            low_memory=False,
        )
    frame = frame[
        frame["Parameter Code"].isin(POLLUTANTS)
        & frame["Sample Duration"].astype(str).str.contains("1 HOUR", na=False)
    ].copy()
    frame["year"] = year
    frame["pollutant"] = frame["Parameter Code"].map(POLLUTANTS)
    frame["site_id"] = (
        frame["State Code"].str.zfill(2)
        + "-"
        + frame["County Code"].str.zfill(3)
        + "-"
        + frame["Site Num"].str.zfill(4)
    )
    # The annual file contains repeated rows for standards/method summaries. Keep
    # the record with the greatest observed-hour count per site/pollutant/year.
    frame = frame.sort_values(
        ["site_id", "pollutant", "year", "Observation Count"],
        ascending=[True, True, True, False],
    ).drop_duplicates(["site_id", "pollutant", "year"], keep="first")
    return frame


def all_years_coverage(frame: pd.DataFrame, minimum_percent: float) -> pd.DataFrame:
    grouped = (
        frame.groupby(
            [
                "CBSA Name",
                "site_id",
                "pollutant",
                "Local Site Name",
                "State Name",
                "County Name",
                "City Name",
                "Latitude",
                "Longitude",
            ],
            dropna=False,
            as_index=False,
        )
        .agg(
            years=("year", "nunique"),
            min_observation_percent=("Observation Percent", "min"),
            mean_observation_percent=("Observation Percent", "mean"),
            min_observation_count=("Observation Count", "min"),
        )
    )
    return grouped[
        (grouped["years"] == len(YEARS))
        & (grouped["min_observation_percent"] >= minimum_percent)
        & grouped["CBSA Name"].notna()
        & (grouped["CBSA Name"].astype(str).str.len() > 0)
    ].copy()


def rank_networks(coverage: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    presence = coverage.pivot_table(
        index=["CBSA Name", "site_id"],
        columns="pollutant",
        values="min_observation_percent",
        aggfunc="max",
    ).reset_index()
    for pollutant in POLLUTANTS.values():
        if pollutant not in presence:
            presence[pollutant] = pd.NA

    pollutant_columns = list(POLLUTANTS.values())
    presence["pollutant_count"] = presence[pollutant_columns].notna().sum(axis=1)
    presence["has_pm25"] = presence["PM2.5"].notna()
    presence["pm25_plus_any_two"] = presence["has_pm25"] & (
        presence[pollutant_columns].notna().sum(axis=1) >= 3
    )
    presence["core_pm25_no2_o3"] = presence[["PM2.5", "NO2", "O3"]].notna().all(axis=1)

    cbsa = (
        presence.groupby("CBSA Name", as_index=False)
        .agg(
            eligible_sites=("site_id", "nunique"),
            pm25_sites=("has_pm25", "sum"),
            pm25_plus_any_two_sites=("pm25_plus_any_two", "sum"),
            core_pm25_no2_o3_sites=("core_pm25_no2_o3", "sum"),
            mean_pollutants_per_site=("pollutant_count", "mean"),
        )
    )
    network_pollutants = (
        coverage.groupby("CBSA Name")["pollutant"]
        .nunique()
        .rename("network_pollutant_count")
        .reset_index()
    )
    cbsa = cbsa.merge(network_pollutants, on="CBSA Name", how="left")
    cbsa["selection_score"] = (
        8 * cbsa["core_pm25_no2_o3_sites"]
        + 4 * cbsa["pm25_plus_any_two_sites"]
        + 2 * cbsa["pm25_sites"]
        + cbsa["eligible_sites"].clip(upper=20)
        + cbsa["network_pollutant_count"]
    )
    cbsa = cbsa.sort_values(
        [
            "selection_score",
            "core_pm25_no2_o3_sites",
            "pm25_plus_any_two_sites",
            "pm25_sites",
        ],
        ascending=False,
    ).reset_index(drop=True)
    return cbsa, presence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data_external/epa_aqs/raw_selection"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data_external/epa_aqs/selection_audit"),
    )
    parser.add_argument("--minimum-percent", type=float, default=50.0)
    args = parser.parse_args()

    frames = [
        read_annual(args.input_dir / f"annual_conc_by_monitor_{year}.zip", year)
        for year in YEARS
    ]
    annual = pd.concat(frames, ignore_index=True)
    coverage = all_years_coverage(annual, args.minimum_percent)
    ranking, site_presence = rank_networks(coverage)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(args.output_dir / "cbsa_ranking.csv", index=False)
    site_presence.to_csv(args.output_dir / "site_pollutant_presence.csv", index=False)
    coverage.to_csv(args.output_dir / "eligible_site_pollutant_coverage.csv", index=False)

    top = ranking.head(20).copy()
    summary = {
        "selection_years": list(YEARS),
        "holdout_year_inspected": False,
        "minimum_observation_percent_each_year": args.minimum_percent,
        "pollutants": list(POLLUTANTS.values()),
        "annual_filtered_rows": len(annual),
        "eligible_site_pollutant_records": len(coverage),
        "candidate_cbsa_count": len(ranking),
        "top_candidates": top.to_dict(orient="records"),
        "selection_warning": (
            "Ranking is a development-data audit, not final network selection. "
            "Hourly data and duplicate-monitor rules must be audited before freezing."
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    columns = [
        "CBSA Name",
        "eligible_sites",
        "pm25_sites",
        "pm25_plus_any_two_sites",
        "core_pm25_no2_o3_sites",
        "network_pollutant_count",
        "mean_pollutants_per_site",
        "selection_score",
    ]
    print(top[columns].to_string(index=False), flush=True)
    print(f"\nOutputs: {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
