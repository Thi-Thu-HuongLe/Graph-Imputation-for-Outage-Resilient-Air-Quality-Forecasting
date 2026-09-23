"""Download immutable US EPA AQS source archives with checksums.

The default ``selection`` stage downloads only site/monitor metadata and annual
monitor summaries for 2021--2024. It intentionally excludes 2025 so that network
selection cannot depend on holdout outcomes.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

BASE_URL = "https://aqs.epa.gov/aqsweb/airdata"
SELECTION_FILES = [
    "aqs_sites.zip",
    "aqs_monitors.zip",
    *(f"annual_conc_by_monitor_{year}.zip" for year in range(2021, 2025)),
]
POLLUTANTS = {
    "88101": "PM2.5",
    "42602": "NO2",
    "44201": "O3",
    "42101": "CO",
    "42401": "SO2",
}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path, retries: int = 3) -> dict[str, object]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")

    if destination.exists():
        return {
            "file": destination.name,
            "url": url,
            "bytes": destination.stat().st_size,
            "sha256": sha256_file(destination),
            "status": "existing_verified",
        }

    request = urllib.request.Request(
        url,
        headers={"User-Agent": "AQ-outage-research/1.0 (academic reproducibility)"},
    )
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            digest = hashlib.sha256()
            byte_count = 0
            with urllib.request.urlopen(request, timeout=120) as response, partial.open(
                "wb"
            ) as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
                    digest.update(chunk)
                    byte_count += len(chunk)
                    if byte_count and byte_count % (25 * 1024 * 1024) < len(chunk):
                        print(
                            f"  {destination.name}: {byte_count / 1024**2:.1f} MiB",
                            flush=True,
                        )
            os.replace(partial, destination)
            return {
                "file": destination.name,
                "url": url,
                "bytes": byte_count,
                "sha256": digest.hexdigest(),
                "status": "downloaded",
            }
        except (OSError, urllib.error.URLError) as error:
            last_error = error
            if partial.exists():
                partial.unlink()
            if attempt < retries:
                print(f"  retry {attempt}/{retries} after: {error}", flush=True)
                time.sleep(2**attempt)
    raise RuntimeError(f"Failed to download {url}") from last_error


def download_api_gzip(
    url: str, destination: Path, retries: int = 3
) -> dict[str, object]:
    """Stream one API response into gzip without interpreting concentration data."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    if destination.exists():
        return {
            "file": destination.name,
            "url": url,
            "bytes": destination.stat().st_size,
            "sha256": sha256_file(destination),
            "status": "existing_verified",
        }

    request = urllib.request.Request(
        url,
        headers={"User-Agent": "AQ-outage-research/1.0 (academic reproducibility)"},
    )
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            raw_bytes = 0
            with urllib.request.urlopen(request, timeout=300) as response, gzip.open(
                partial, "wb", compresslevel=6
            ) as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
                    raw_bytes += len(chunk)
                    if raw_bytes and raw_bytes % (25 * 1024 * 1024) < len(chunk):
                        print(
                            f"  {destination.name}: {raw_bytes / 1024**2:.1f} MiB raw",
                            flush=True,
                        )
            os.replace(partial, destination)
            return {
                "file": destination.name,
                "url": url,
                "compressed_bytes": destination.stat().st_size,
                "raw_bytes": raw_bytes,
                "sha256": sha256_file(destination),
                "status": "downloaded",
            }
        except (OSError, urllib.error.URLError) as error:
            last_error = error
            if partial.exists():
                partial.unlink()
            if attempt < retries:
                print(f"  retry {attempt}/{retries} after: {error}", flush=True)
                time.sleep(2**attempt)
    raise RuntimeError(f"Failed to download {url}") from last_error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=["selection", "development_api", "holdout_api"],
        default="selection",
        help="Select metadata, development years, or a sealed holdout year.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    parser.add_argument("--years", nargs="+", type=int)
    parser.add_argument("--pollutants", nargs="+", choices=sorted(POLLUTANTS))
    parser.add_argument("--state", default="49", help="Two-digit state FIPS code.")
    parser.add_argument("--county", default="035", help="Three-digit county FIPS code.")
    parser.add_argument(
        "--api-email", default=os.environ.get("AQS_API_EMAIL", "test@aqs.api")
    )
    parser.add_argument("--api-key", default=os.environ.get("AQS_API_KEY", "test"))
    parser.add_argument(
        "--protocol-file",
        type=Path,
        default=Path("journal_protocol/unseen_holdout_protocol.json"),
    )
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = {
            "selection": Path("data_external/epa_aqs/raw_selection"),
            "development_api": Path("data_external/epa_aqs/raw_development_api"),
            "holdout_api": Path("data_external/epa_aqs/raw_holdout_sealed"),
        }[args.stage]

    if args.stage == "selection":
        files = SELECTION_FILES
        years: list[int] = []
        pollutants: list[str] = []
    else:
        files = []
        years = args.years or (
            list(range(2021, 2025)) if args.stage == "development_api" else [2025]
        )
        pollutants = args.pollutants or list(POLLUTANTS)
        if args.stage == "development_api" and any(year >= 2025 for year in years):
            raise ValueError("development_api cannot download the 2025 holdout")
        if args.stage == "holdout_api" and years != [2025]:
            raise ValueError("holdout_api is restricted to the sealed 2025 holdout")

    records: list[dict[str, object]] = []
    print(f"EPA AQS download stage: {args.stage}", flush=True)
    print(f"Destination: {args.output_dir.resolve()}", flush=True)
    for index, name in enumerate(files, start=1):
        print(f"[{index}/{len(files)}] {name}", flush=True)
        records.append(download(f"{BASE_URL}/{name}", args.output_dir / name))

    api_requests = [(year, pollutant) for year in years for pollutant in pollutants]
    for index, (year, pollutant) in enumerate(api_requests, start=1):
        label = POLLUTANTS[pollutant]
        name = (
            f"sampleData_state-{args.state}_county-{args.county}_"
            f"{pollutant}-{label}_duration-1H_{year}.json.gz"
        )
        query = urllib.parse.urlencode(
            {
                "email": args.api_email,
                "key": args.api_key,
                "param": pollutant,
                "bdate": f"{year}0101",
                "edate": f"{year}1231",
                "state": args.state,
                "county": args.county,
                "duration": "1",
            }
        )
        url = f"https://aqs.epa.gov/data/api/sampleData/byCounty?{query}"
        print(
            f"[{index}/{len(api_requests)}] {year} {label} "
            f"(state={args.state}, county={args.county})",
            flush=True,
        )
        record = download_api_gzip(url, args.output_dir / name)
        # Do not retain credentials in a publication artifact.
        record["url"] = (
            "https://aqs.epa.gov/data/api/sampleData/byCounty?"
            f"param={pollutant}&bdate={year}0101&edate={year}1231&"
            f"state={args.state}&county={args.county}&duration=1&credentials=REDACTED"
        )
        record["year"] = year
        record["parameter_code"] = pollutant
        record["pollutant"] = label
        record["sample_duration_code"] = "1"
        records.append(record)
        if index < len(api_requests):
            time.sleep(6.5)

    manifest = {
        "source": "US EPA Air Quality System (AQS) AirData",
        "source_page": f"{BASE_URL}/download_files.html",
        "stage": args.stage,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "holdout_data_included": args.stage == "holdout_api",
        "sealed_without_content_audit": args.stage == "holdout_api",
        "state_fips": args.state if args.stage != "selection" else None,
        "county_fips": args.county if args.stage != "selection" else None,
        "files": records,
    }
    if args.stage == "holdout_api":
        manifest["protocol_file_at_download"] = str(args.protocol_file.as_posix())
        manifest["protocol_sha256_at_download"] = sha256_file(args.protocol_file)
    manifest_path = args.output_dir / "download_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Manifest: {manifest_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
