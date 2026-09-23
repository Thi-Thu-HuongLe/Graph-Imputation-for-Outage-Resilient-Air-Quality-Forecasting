"""Validate the EPA AQS development download against its manifest and API metadata."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data_external/epa_aqs/raw_development_api"),
    )
    args = parser.parse_args()

    manifest_path = args.input_dir / "download_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["stage"] != "development_api" or manifest["holdout_data_included"]:
        raise ValueError("Expected an unsealed development_api manifest")

    summaries: list[dict[str, object]] = []
    errors: list[str] = []
    for index, record in enumerate(manifest["files"], start=1):
        path = args.input_dir / record["file"]
        actual_hash = sha256_file(path)
        if actual_hash != record["sha256"]:
            errors.append(f"SHA-256 mismatch: {path.name}")

        with gzip.open(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        header = payload.get("Header", [])
        data = payload.get("Data", [])
        status = header[0].get("status") if header else None
        header_rows = header[0].get("rows") if header else None
        expected_parameter = str(record["parameter_code"])
        parameter_counts = Counter(str(row.get("parameter_code")) for row in data)
        duration_counts = Counter(str(row.get("sample_duration_code")) for row in data)
        state_counts = Counter(str(row.get("state_code")) for row in data)
        county_counts = Counter(str(row.get("county_code")) for row in data)
        unit_counts = Counter(str(row.get("units_of_measure")) for row in data)
        site_counts = Counter(str(row.get("site_number")) for row in data)

        if status != "Success":
            errors.append(f"API status is {status!r}: {path.name}")
        if header_rows is not None and int(header_rows) != len(data):
            errors.append(f"Header/data row mismatch: {path.name}")
        if set(parameter_counts) != {expected_parameter}:
            errors.append(f"Unexpected parameter code: {path.name}: {parameter_counts}")
        if set(duration_counts) != {"1"}:
            errors.append(f"Unexpected sample duration: {path.name}: {duration_counts}")
        if set(state_counts) != {manifest["state_fips"]}:
            errors.append(f"Unexpected state code: {path.name}: {state_counts}")
        if set(county_counts) != {manifest["county_fips"]}:
            errors.append(f"Unexpected county code: {path.name}: {county_counts}")

        summaries.append(
            {
                "file": path.name,
                "year": int(record["year"]),
                "pollutant": record["pollutant"],
                "parameter_code": expected_parameter,
                "rows": len(data),
                "site_count": len(site_counts),
                "sites": sorted(site_counts),
                "units": dict(sorted(unit_counts.items())),
                "sha256": actual_hash,
                "status": "PASS" if not any(path.name in error for error in errors) else "FAIL",
            }
        )
        print(
            f"[{index}/{len(manifest['files'])}] {path.name}: "
            f"rows={len(data):,}, sites={len(site_counts)}, status={status}",
            flush=True,
        )

    output = {
        "validation_status": "PASS" if not errors else "FAIL",
        "manifest": str(manifest_path.as_posix()),
        "file_count": len(summaries),
        "total_rows": sum(int(item["rows"]) for item in summaries),
        "errors": errors,
        "files": summaries,
    }
    output_path = args.input_dir / "validation_summary.json"
    output_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Validation: {output['validation_status']}", flush=True)
    print(f"Summary: {output_path.resolve()}", flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
