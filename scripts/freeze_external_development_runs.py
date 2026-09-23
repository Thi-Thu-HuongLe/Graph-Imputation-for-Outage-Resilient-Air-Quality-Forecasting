"""Freeze all external development runs without reopening candidate selection."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from aqriskformer.utils import read_json, sha256_file, utc_now, write_json
from scripts.freeze_journal_candidate import verify_run

CANDIDATE = "adaptive_graph_impute_tcn"
LEARNED = (
    "local_tcn",
    "local_capacity_residual",
    "impute_then_local_tcn",
    "dcrnn",
    "graph_wavenet",
)
DETERMINISTIC = ("persistence", "seasonal_naive_24h")
SEEDS = (42, 123, 2026, 3407, 7777)


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT)).replace("\\", "/")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-root",
        type=Path,
        default=ROOT / "experiment_protocol/results_external/development/adaptive_5seed",
    )
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=ROOT / "experiment_protocol/results_external/development/base_full",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "journal_protocol/external_development_runs_freeze.json",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite external run freeze: {args.output}")
    candidate_root = args.candidate_root.resolve()
    baseline_root = args.baseline_root.resolve()
    for root in (candidate_root, baseline_root):
        launch = read_json(root / "launch_manifest.json")
        status = read_json(root / "status.json")
        if launch.get("holdout_accessed") is not False:
            raise RuntimeError(f"Development launch reports holdout access: {root}")
        if status.get("state") != "complete" or status.get("holdout_accessed") is not False:
            raise RuntimeError(f"External development run is incomplete: {root}")

    analysis_root = candidate_root / "analysis"
    analysis_manifest = read_json(analysis_root / "analysis_manifest.json")
    decision = read_json(analysis_root / "development_decision.json")
    if analysis_manifest.get("candidate_selection_reopened") is not False:
        raise RuntimeError("External analysis does not preserve the closed selection gate")
    if decision.get("freeze_recommendation") != "candidate_already_frozen_do_not_reselect":
        raise RuntimeError("External diagnostic attempted to reopen candidate selection")

    run_hashes: dict[str, str] = {}
    for seed in SEEDS:
        run_hashes.update(
            verify_run(candidate_root / "runs" / CANDIDATE / f"seed_{seed}", CANDIDATE, seed)
        )
    for model in LEARNED:
        for seed in SEEDS:
            run_hashes.update(
                verify_run(baseline_root / "runs" / model / f"seed_{seed}", model, seed)
            )
    for model in DETERMINISTIC:
        run_hashes.update(verify_run(baseline_root / "runs" / model, model, None))

    protocol = ROOT / "journal_protocol/external_comparison_protocol.json"
    prepared = ROOT / read_json(protocol)["prepared_development"]
    source_paths = [
        ROOT / "journal_protocol/external_replication_selection.json",
        ROOT / "journal_protocol/external_unseen_holdout_protocol.json",
        ROOT / "journal_protocol/external_development_preprocessing_protocol.json",
        protocol,
        ROOT / "journal_protocol/candidate_selection_freeze.json",
        ROOT / "scripts/run_external_development.py",
        ROOT / "scripts/analyze_external_development.py",
        ROOT / "scripts/run_journal_development.py",
        ROOT / "scripts/analyze_journal_development.py",
        ROOT / "src/aqriskformer/outage_models.py",
        ROOT / "src/aqriskformer/outage_data.py",
        ROOT / "src/aqriskformer/outage_evaluation.py",
        ROOT / "src/aqriskformer/statistics.py",
        Path(__file__).resolve(),
    ]
    salt_freeze = ROOT / "journal_protocol/candidate_selection_freeze.json"
    record = {
        "status": "EXTERNAL_DEVELOPMENT_RUNS_FROZEN_2025_NOT_DOWNLOADED",
        "created_utc": utc_now(),
        "candidate": CANDIDATE,
        "candidate_selection_source": rel(salt_freeze),
        "candidate_selection_source_sha256": sha256_file(salt_freeze),
        "candidate_selected_on_external_network": False,
        "comparators": [*LEARNED, *DETERMINISTIC],
        "seeds": list(SEEDS),
        "checkpoint_roots": {
            "candidate": rel(candidate_root / "runs"),
            "comparators": rel(baseline_root / "runs"),
        },
        "analysis": {
            "path": rel(analysis_root),
            "manifest_sha256": sha256_file(analysis_root / "analysis_manifest.json"),
            "decision_sha256": sha256_file(analysis_root / "development_decision.json"),
            "role": "diagnostic_only",
        },
        "prepared_development": rel(prepared),
        "prepared_development_sha256": sha256_file(prepared),
        "source_sha256": {rel(path): sha256_file(path) for path in source_paths},
        "run_artifact_sha256": run_hashes,
        "external_holdout": {
            "year": 2025,
            "downloaded": False,
            "content_parsed": False,
            "metrics_computed": False,
        },
        "holdout_download_authorized": True,
        "holdout_evaluation_authorized": False,
        "next_gate": (
            "Download Clark County 2025 into the sealed external directory without parsing "
            "content, then create the final protocol and execution freeze."
        ),
    }
    write_json(args.output, record)
    print(f"External development runs frozen: {args.output.resolve()}")
    print(f"SHA-256: {sha256_file(args.output)}")
    print("External 2025 may now be downloaded and sealed, but not parsed.")


if __name__ == "__main__":
    main()
