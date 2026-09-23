from __future__ import annotations

from scripts.select_external_replication_network import select_network


def row(cbsa: str, site: str, **coverage: str) -> dict[str, str]:
    item = {
        "CBSA Name": cbsa,
        "site_id": site,
        "PM2.5": "",
        "NO2": "",
        "O3": "",
        "CO": "",
        "SO2": "",
    }
    item.update(coverage)
    return item


def test_selection_excludes_salt_lake_and_multi_county_networks() -> None:
    rows = [
        row(
            "Salt Lake City, UT",
            f"49-035-{index:04d}",
            **{"PM2.5": "90", "NO2": "90", "O3": "90", "CO": "90", "SO2": "90"},
        )
        for index in range(7)
    ]
    rows += [
        row(
            "Multi County",
            f"01-{county}-{index:04d}",
            **{"PM2.5": "90", "NO2": "90", "O3": "90", "CO": "90", "SO2": "90"},
        )
        for index, county in enumerate(["001", "003", "001", "003", "001", "003", "001"])
    ]
    rows += [
        row(
            "External Single County",
            f"32-003-{index:04d}",
            **{"PM2.5": "90", "NO2": "90", "O3": "90", "CO": "90", "SO2": "90"},
        )
        for index in range(7)
    ]

    selected, stations, candidates = select_network(rows, 7)

    assert selected["cbsa"] == "External Single County"
    assert len(stations) == 7
    assert [item["cbsa"] for item in candidates] == ["External Single County"]


def test_station_tie_breaking_prefers_pollutant_coverage_then_site_id() -> None:
    rows = [
        row(
            "External",
            f"32-003-{index:04d}",
            **{
                "PM2.5": "90",
                "NO2": "90" if index < 3 else "",
                "O3": "90",
                "CO": "90" if index == 0 else "",
                "SO2": "90" if index == 0 else "",
            },
        )
        for index in range(8)
    ]
    selected, stations, _ = select_network(rows, 7)

    assert selected["cbsa"] == "External"
    assert stations[0]["site_id"] == "32-003-0000"
    assert stations[1]["site_id"] == "32-003-0001"
    assert stations[-1]["site_id"] == "32-003-0006"
