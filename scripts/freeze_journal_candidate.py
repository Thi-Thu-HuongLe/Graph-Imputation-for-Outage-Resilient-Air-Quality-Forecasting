"""Freeze the validation-selected candidate while keeping the 2025 holdout sealed."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aqriskformer.utils import read_json, sha256_file, utc_now, write_json

CANDIDATE_ROOT = ROOT / (
    "experiment_protocol/results_journal/development/adaptive_graph_impute_5seed"
)
BASELINE_ROOT = ROOT / "experiment_protocol/results_journal/development/full"
ANALYSIS_ROOT = CANDIDATE_ROOT / "analysis"
OUTPUT = ROOT / "journal_protocol/candidate_selection_freeze.json"
SEEDS = (42, 123, 2026, 3407, 7777)
CANDIDATE = "adaptive_graph_impute_tcn"
LEARNED_COMPARATORS = (
    "local_tcn",
    "local_capacity_residual",
    "impute_then_local_tcn",
    "dcrnn",
    "graph_wavenet",
)
DETERMINISTIC_COMPARATORS = ("persistence", "seasonal_naive_24h")


def verify_hashes(base: Path, values: dict[str, str]) -> None:
    for name, expected in values.items():
        path = base / name
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"Frozen hash mismatch: {path}")


def verify_run(directory: Path, model: str, seed: int | None) -> dict[str, str]:
    marker_path = directory / "complete.json"
    marker = read_json(marker_path)
    if marker.get("state") != "complete" or marker.get("model") != model:
        raise RuntimeError(f"Incomplete or mismatched run: {directory}")
    if seed is not None and int(marker.get("seed", -1)) != seed:
        raise RuntimeError(f"Seed mismatch: {directory}")
    verify_hashes(directory, marker["artifact_hashes"])
    required = ["calibration.npz"]
    required.append("best.pt" if seed is not None else "training_residual_sigma.npz")
    for name in required:
        if name not in marker["artifact_hashes"]:
            raise RuntimeError(f"Run marker does not freeze {name}: {directory}")
    return {
        str(marker_path.relative_to(ROOT)): sha256_file(marker_path),
        **{
            str((directory / name).relative_to(ROOT)): digest
            for name, digest in marker["artifact_hashes"].items()
        },
    }


def main() -> None:
    if OUTPUT.exists():
        existing = read_json(OUTPUT)
        if existing.get("status") != "CANDIDATE_SELECTION_FROZEN_HOLDOUT_UNOPENED":
            raise RuntimeError(f"Unexpected existing freeze record: {OUTPUT}")
        print(f"Candidate selection is already frozen: {OUTPUT}")
        print(f"SHA-256: {sha256_file(OUTPUT)}")
        return

    candidate_launch = read_json(CANDIDATE_ROOT / "launch_manifest.json")
    candidate_status = read_json(CANDIDATE_ROOT / "status.json")
    if candidate_launch.get("holdout_accessed") is not False:
        raise RuntimeError("Candidate run does not preserve the sealed-holdout invariant")
    if candidate_status.get("state") != "complete" or candidate_status.get(
        "holdout_accessed"
    ) is not False:
        raise RuntimeError("Candidate run is incomplete or reports holdout access")

    analysis_manifest = read_json(ANALYSIS_ROOT / "analysis_manifest.json")
    if analysis_manifest.get("holdout_accessed") is not False:
        raise RuntimeError("Development analysis reports holdout access")
    verify_hashes(ROOT, analysis_manifest["input_sha256"])
    verify_hashes(ANALYSIS_ROOT, analysis_manifest["output_sha256"])
    decision = read_json(ANALYSIS_ROOT / "development_decision.json")
    if decision.get("freeze_recommendation") != "freeze_candidate_then_open_holdout_once":
        raise RuntimeError("Development analysis does not recommend candidate freeze")

    holdout_seal = read_json(ROOT / "journal_protocol/holdout_seal_status.json")
    if holdout_seal.get("status") != "DOWNLOADED_AND_SEALED_NOT_PARSED":
        raise RuntimeError("Holdout seal status is not intact")
    holdout_manifest = ROOT / holdout_seal["holdout_manifest"]
    if sha256_file(holdout_manifest) != holdout_seal["holdout_manifest_sha256"]:
        raise RuntimeError("Sealed holdout manifest hash differs from its seal record")
    if any(
        holdout_seal.get(key) is not False
        for key in (
            "gzip_or_json_content_opened",
            "concentration_values_inspected",
            "test_metrics_computed",
        )
    ):
        raise RuntimeError("Holdout seal record indicates prior content access")

    run_hashes: dict[str, str] = {}
    for seed in SEEDS:
        run_hashes.update(
            verify_run(
                CANDIDATE_ROOT / "runs" / CANDIDATE / f"seed_{seed}",
                CANDIDATE,
                seed,
            )
        )
    for model in LEARNED_COMPARATORS:
        for seed in SEEDS:
            run_hashes.update(
                verify_run(
                    BASELINE_ROOT / "runs" / model / f"seed_{seed}", model, seed
                )
            )
    for model in DETERMINISTIC_COMPARATORS:
        run_hashes.update(verify_run(BASELINE_ROOT / "runs" / model, model, None))

    source_paths = (
        ROOT / "journal_protocol/unseen_holdout_protocol.json",
        ROOT / "journal_protocol/development_preprocessing_protocol.json",
        ROOT / "journal_protocol/comparison_protocol.json",
        ROOT / "journal_protocol/holdout_seal_status.json",
        ROOT / "src/aqriskformer/outage_models.py",
        ROOT / "src/aqriskformer/outage_data.py",
        ROOT / "src/aqriskformer/outage_evaluation.py",
        ROOT / "src/aqriskformer/development_analysis.py",
        ROOT / "src/aqriskformer/statistics.py",
        ROOT / "scripts/run_journal_development.py",
        ROOT / "scripts/analyze_journal_development.py",
        Path(__file__).resolve(),
    )
    freeze = {
        "status": "CANDIDATE_SELECTION_FROZEN_HOLDOUT_UNOPENED",
        "created_utc": utc_now(),
        "scientific_question": (
            "Whether a validation-selected adaptive spatial imputation residual improves "
            "probabilistic multi-pollutant forecasting during target-station outages."
        ),
        "selected_candidate": CANDIDATE,
        "selection_basis": {
            "development_period": "EPA AQS Salt Lake County 2021-2024 only",
            "seeds": list(SEEDS),
            "primary_conditions": [
                "each_station_trailing_6h_outage",
                "each_station_trailing_24h_outage",
            ],
            "primary_endpoints": ["mase", "q95_brier"],
            "decision_sha256": sha256_file(
                ANALYSIS_ROOT / "development_decision.json"
            ),
            "analysis_manifest_sha256": sha256_file(
                ANALYSIS_ROOT / "analysis_manifest.json"
            ),
            "development_summary": decision[
                "candidate_vs_frozen_imputation_baseline"
            ],
        },
        "final_comparators": {
            "learned": list(LEARNED_COMPARATORS),
            "deterministic": list(DETERMINISTIC_COMPARATORS),
        },
        "calibration_lock": {
            "method": (
                "per-pollutant multiplicative Gaussian sigma factors fitted by masked "
                "NLL on clean 2024 validation predictions"
            ),
            "factors": "the calibration.npz file frozen separately for every model/seed",
            "duration_specific_calibration_rejected": True,
            "reason": (
                "post-selection development diagnostic worsened Q95 Brier and did not "
                "consistently improve the 80% coverage error"
            ),
            "test_labels_used": False,
        },
        "statistics_lock": {
            "temporal_block_length_hours": 168,
            "bootstrap_resamples": 10000,
            "primary_family": (
                "candidate versus every final comparator for MASE and Q95 Brier under "
                "6 h and 24 h trailing station outages"
            ),
            "multiplicity": "Holm correction over the complete primary family",
            "seeds": "optimization variability; predictions averaged by aligned origin before temporal inference",
        },
        "claim_boundary": decision["claim_boundary"],
        "holdout": {
            "year": 2025,
            "manifest": str(holdout_manifest.relative_to(ROOT)),
            "manifest_sha256": sha256_file(holdout_manifest),
            "content_parsed": False,
            "metrics_computed": False,
        },
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256_file(path) for path in source_paths
        },
        "run_artifact_sha256": run_hashes,
        "next_gate": (
            "Implement, test, and hash-freeze the holdout preparation, inference, and "
            "reporting code before any 2025 gzip/JSON content is opened."
        ),
        "holdout_evaluation_authorized": False,
    }
    write_json(OUTPUT, freeze)
    print(f"Candidate selection frozen: {OUTPUT}")
    print(f"SHA-256: {sha256_file(OUTPUT)}")
    print("Holdout remains sealed; final execution code is the next gate.")


if __name__ == "__main__":
    main()
