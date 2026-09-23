"""Analyze external-network development runs without reopening model selection."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from aqriskformer.utils import read_json, sha256_file, write_json
from scripts import analyze_journal_development as analysis

EXTERNAL_PROTOCOL = ROOT / "journal_protocol/external_comparison_protocol.json"


def requested_paths() -> tuple[Path, Path]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--candidate-root", type=Path, default=analysis.DEFAULT_CANDIDATE_ROOT)
    parser.add_argument("--output", type=Path)
    args, _ = parser.parse_known_args()
    candidate = args.candidate_root.resolve()
    return candidate, args.output.resolve() if args.output else candidate / "analysis"


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT)).replace("\\", "/")


def rel_to_output(path: Path, output: Path) -> str:
    return str(path.resolve().relative_to(output.resolve())).replace("\\", "/")


def mark_external_diagnostic(output: Path) -> None:
    decision_path = output / "development_decision.json"
    decision = read_json(decision_path)
    decision["salt_lake_candidate_already_frozen"] = True
    decision["external_network_model_selection_permitted"] = False
    decision["external_development_diagnostic_outcome"] = decision.get(
        "freeze_recommendation"
    )
    decision["freeze_recommendation"] = "candidate_already_frozen_do_not_reselect"
    decision["claim_boundary"] = (
        "External 2024 results are development diagnostics only. The Salt Lake-selected "
        "candidate, calibration method, endpoints, and 2025 external evaluation protocol "
        "must not be changed from these results."
    )
    write_json(decision_path, decision)

    manifest_path = output / "analysis_manifest.json"
    manifest = read_json(manifest_path)
    inputs = dict(manifest["input_sha256"])
    inputs.pop("journal_protocol/comparison_protocol.json", None)
    inputs[rel(EXTERNAL_PROTOCOL)] = sha256_file(EXTERNAL_PROTOCOL)
    inputs[rel(Path(__file__).resolve())] = sha256_file(Path(__file__).resolve())
    manifest["input_sha256"] = dict(sorted(inputs.items()))
    manifest["external_geographic_replication"] = True
    manifest["candidate_selection_reopened"] = False
    output_files = [
        output / "primary_paired_tests.csv",
        output / "per_seed_effects.csv",
        output / "calibration_diagnostics_per_seed.csv",
        output / "calibration_diagnostics_summary.csv",
        decision_path,
    ]
    manifest["output_sha256"] = {
        rel_to_output(path, output): sha256_file(path) for path in output_files
    }
    write_json(manifest_path, manifest)


def main() -> None:
    _, output = requested_paths()
    analysis.main()
    mark_external_diagnostic(output)
    print("External development analysis marked diagnostic-only; model selection remains closed.")


if __name__ == "__main__":
    main()
