"""Analyze and report every prespecified Clark County external holdout result."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import analyze_journal_holdout as analysis

analysis.PROTOCOL_PATH = ROOT / "journal_protocol/external_final_holdout_protocol.json"
analysis.FREEZE_PATH = ROOT / "journal_protocol/external_final_holdout_execution_freeze.json"


if __name__ == "__main__":
    analysis.main()
