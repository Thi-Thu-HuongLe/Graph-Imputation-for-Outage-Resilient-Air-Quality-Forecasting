"""Build the locked EPA AQS 2021--2024 development artifact and audit tables."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aqriskformer.data import _calendar_features
from aqriskformer.epa_aqs import prepare_epa_development
from aqriskformer.utils import read_json, sha256_file, utc_now, write_json

DEFAULT_PARENT = ROOT / "journal_protocol/unseen_holdout_protocol.json"
DEFAULT_PREPROCESSING = ROOT / "journal_protocol/development_preprocessing_protocol.json"
DEFAULT_RAW = ROOT / "data_external/epa_aqs/raw_development_api"


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-protocol", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--preprocessing-protocol", type=Path, default=DEFAULT_PREPROCESSING)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    args = parser.parse_args()
    specification = read_json(args.preprocessing_protocol)
    prepared_path = project_path(specification["prepared_output"])
    report_path = project_path(specification["preprocessing_report"])
    availability_path = project_path(specification["availability_table"])

    prepared, report, availability = prepare_epa_development(
        args.parent_protocol.resolve(),
        args.preprocessing_protocol.resolve(),
        args.raw_dir.resolve(),
    )
    offset = float(
        specification["time_axis"].get("local_standard_utc_offset_hours", -7.0)
    )
    if not -14.0 <= offset <= 14.0:
        raise ValueError(f"Invalid fixed UTC offset: {offset}")
    prepared.calendar = _calendar_features(
        prepared.timestamps + pd.to_timedelta(offset, unit="h")
    )
    if specification.get("prepared_dataset_name"):
        prepared.name = str(specification["prepared_dataset_name"])
    report["calendar_clock"] = {
        "kind": "fixed local standard time",
        "utc_offset_hours": offset,
    }
    prepared.save(prepared_path)
    availability_path.parent.mkdir(parents=True, exist_ok=True)
    availability.to_csv(availability_path, index=False)
    report.update(
        completed_utc=utc_now(),
        prepared_file=str(prepared_path.relative_to(ROOT)),
        prepared_sha256=sha256_file(prepared_path),
        availability_file=str(availability_path.relative_to(ROOT)),
        availability_sha256=sha256_file(availability_path),
    )
    write_json(report_path, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "prepared": str(prepared_path),
                "prepared_sha256": report["prepared_sha256"],
                "shape": report["shape"],
                "split_timestamps_utc": report["split_timestamps_utc"],
                "observed_fraction_by_pollutant": report["observed_fraction_by_pollutant"],
                "report": str(report_path),
                "availability": str(availability_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
