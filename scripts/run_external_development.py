"""Run the locked Las Vegas development protocol in isolated output paths."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import run_journal_development as runner

runner.PROTOCOL_PATH = ROOT / "journal_protocol/external_comparison_protocol.json"
runner.DEFAULT_OUTPUT = ROOT / "experiment_protocol/results_external/development"


if __name__ == "__main__":
    runner.main()
