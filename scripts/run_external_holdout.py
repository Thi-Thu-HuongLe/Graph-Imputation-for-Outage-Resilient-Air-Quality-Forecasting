"""Execute the frozen Clark County 2025 external replication."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from aqriskformer.data import _calendar_features
from aqriskformer.epa_aqs_holdout import prepare_epa_holdout as prepare_locked_holdout
from aqriskformer.utils import read_json
from scripts import run_journal_holdout as runner

runner.PROTOCOL_PATH = ROOT / "journal_protocol/external_final_holdout_protocol.json"
runner.FREEZE_PATH = ROOT / "journal_protocol/external_final_holdout_execution_freeze.json"
runner.PREPROCESSING_PATH = (
    ROOT / "journal_protocol/external_development_preprocessing_protocol.json"
)


def runtime_protocol(final: dict[str, object]) -> dict[str, object]:
    protocol = read_json(ROOT / "journal_protocol/external_comparison_protocol.json")
    for name in (
        "lookback",
        "horizon",
        "reported_horizons",
        "test_stride",
        "fill_limit",
        "corruption_seed",
    ):
        protocol[name] = final["forecast"][name]
    return protocol


def prepare_external_holdout(*args, **kwargs):
    prepared, report, availability = prepare_locked_holdout(*args, **kwargs)
    protocol = read_json(args[0] if args else kwargs["protocol_path"])
    offset = float(protocol["data"]["local_standard_utc_offset_hours"])
    prepared.calendar = _calendar_features(
        prepared.timestamps + pd.to_timedelta(offset, unit="h")
    )
    prepared.name = str(protocol["data"]["prepared_dataset_name"])
    report["calendar_clock"] = {
        "kind": "fixed local standard time",
        "utc_offset_hours": offset,
    }
    return prepared, report, availability


runner.runtime_protocol = runtime_protocol
runner.prepare_epa_holdout = prepare_external_holdout


if __name__ == "__main__":
    runner.main()
