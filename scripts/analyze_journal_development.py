"""Analyze the selected development candidate without opening the sealed holdout."""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aqriskformer.development_analysis import (
    apply_pollutant_scale,
    finite_column_mean,
    hourly_loss_grid,
    load_trailing_prediction,
    origin_macro_losses,
    verify_paired_predictions,
)
from aqriskformer.outage_data import load_outage_data
from aqriskformer.outage_evaluation import (
    fit_multiplicative_sigma_calibration,
    outage_metrics,
)
from aqriskformer.statistics import (
    diebold_mariano,
    holm_adjust,
    paired_moving_block_bootstrap,
)
from aqriskformer.utils import read_json, sha256_file, utc_now, write_json

DEFAULT_CANDIDATE_ROOT = ROOT / (
    "experiment_protocol/results_journal/development/adaptive_graph_impute_5seed"
)
DEFAULT_BASELINE_ROOT = ROOT / "experiment_protocol/results_journal/development/full"
MODEL_CANDIDATE = "adaptive_graph_impute_tcn"
COMPARATORS = ("impute_then_local_tcn", "dcrnn", "graph_wavenet")
DURATIONS = (6, 24)
ENDPOINTS = ("mase", "q95_brier")


def log(message: str) -> None:
    print(f"[{utc_now()}] {message}", flush=True)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        frame.to_csv(temporary, index=False)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def run_dir(root: Path, model: str, seed: int) -> Path:
    return root / "runs" / model / f"seed_{seed}"


def prediction_path(root: Path, model: str, seed: int) -> Path:
    path = run_dir(root, model, seed) / "trailing_validation_predictions.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def clean_factors(root: Path, model: str, seed: int) -> np.ndarray:
    path = run_dir(root, model, seed) / "calibration.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"factors"}:
            raise ValueError(f"Unexpected calibration schema: {path}")
        return archive["factors"].copy()


def calibrated_prediction(
    root: Path, model: str, seed: int, duration: int
) -> dict[str, np.ndarray]:
    prediction = load_trailing_prediction(prediction_path(root, model, seed), duration)
    return apply_pollutant_scale(prediction, clean_factors(root, model, seed))


def paired_primary_tests(
    candidate_root: Path,
    baseline_root: Path,
    data,
    seeds: list[int],
    block_length: int,
    resamples: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    primary_rows: list[dict[str, object]] = []
    seed_rows: list[dict[str, object]] = []
    for comparator_index, comparator in enumerate(COMPARATORS):
        for duration in DURATIONS:
            candidate_by_endpoint: dict[str, list[np.ndarray]] = {key: [] for key in ENDPOINTS}
            baseline_by_endpoint: dict[str, list[np.ndarray]] = {key: [] for key in ENDPOINTS}
            for seed in seeds:
                candidate = calibrated_prediction(
                    candidate_root, MODEL_CANDIDATE, seed, duration
                )
                baseline = calibrated_prediction(baseline_root, comparator, seed, duration)
                verify_paired_predictions(candidate, baseline)
                candidate_losses = origin_macro_losses(candidate, data)
                baseline_losses = origin_macro_losses(baseline, data)
                for endpoint in ENDPOINTS:
                    candidate_grid = hourly_loss_grid(
                        candidate["origins"], candidate_losses[endpoint]
                    )
                    baseline_grid = hourly_loss_grid(
                        baseline["origins"], baseline_losses[endpoint]
                    )
                    candidate_by_endpoint[endpoint].append(candidate_grid)
                    baseline_by_endpoint[endpoint].append(baseline_grid)
                    valid = np.isfinite(candidate_grid) & np.isfinite(baseline_grid)
                    difference = candidate_grid[valid] - baseline_grid[valid]
                    baseline_mean = float(baseline_grid[valid].mean())
                    seed_rows.append(
                        {
                            "candidate": MODEL_CANDIDATE,
                            "comparator": comparator,
                            "duration_hours": duration,
                            "endpoint": endpoint,
                            "seed": seed,
                            "paired_origins": int(valid.sum()),
                            "candidate_mean": float(candidate_grid[valid].mean()),
                            "comparator_mean": baseline_mean,
                            "mean_difference_candidate_minus_comparator": float(
                                difference.mean()
                            ),
                            "relative_benefit_percent": float(
                                -100.0 * difference.mean() / baseline_mean
                            ),
                            "candidate_win": bool(difference.mean() < 0),
                        }
                    )
            for endpoint_index, endpoint in enumerate(ENDPOINTS):
                candidate_mean = finite_column_mean(candidate_by_endpoint[endpoint])
                baseline_mean = finite_column_mean(baseline_by_endpoint[endpoint])
                bootstrap = paired_moving_block_bootstrap(
                    candidate_mean,
                    baseline_mean,
                    block_length=block_length,
                    resamples=resamples,
                    seed=81317 + 100 * comparator_index + 10 * duration + endpoint_index,
                )
                dm = diebold_mariano(
                    candidate_mean,
                    baseline_mean,
                    horizon=1,
                    newey_west_lag=block_length - 1,
                )
                valid = np.isfinite(candidate_mean) & np.isfinite(baseline_mean)
                comparator_mean = float(baseline_mean[valid].mean())
                primary_rows.append(
                    {
                        "candidate": MODEL_CANDIDATE,
                        "comparator": comparator,
                        "condition": f"each_station_trailing_{duration}h_outage",
                        "duration_hours": duration,
                        "endpoint": endpoint,
                        "seeds_averaged_per_origin": len(seeds),
                        "paired_origins": int(valid.sum()),
                        "candidate_mean": float(candidate_mean[valid].mean()),
                        "comparator_mean": comparator_mean,
                        "mean_difference_candidate_minus_comparator": bootstrap[
                            "mean_difference"
                        ],
                        "relative_benefit_percent": float(
                            -100.0 * bootstrap["mean_difference"] / comparator_mean
                        ),
                        "bootstrap_ci95_lower": bootstrap["ci95_lower"],
                        "bootstrap_ci95_upper": bootstrap["ci95_upper"],
                        "candidate_better_ci95": bool(bootstrap["ci95_upper"] < 0),
                        "dm_statistic": dm["statistic"],
                        "dm_p_value": dm["p_value"],
                    }
                )
                log(f"paired test complete: {comparator}, {duration} h, {endpoint}")

    primary = pd.DataFrame(primary_rows)
    primary["dm_p_holm"] = holm_adjust(primary["dm_p_value"].to_numpy())
    primary["dm_significant_holm_0_05"] = primary["dm_p_holm"] < 0.05
    per_seed = pd.DataFrame(seed_rows)
    return primary, per_seed


def calibration_diagnostics(
    candidate_root: Path,
    baseline_root: Path,
    data,
    seeds: list[int],
    bounds: tuple[float, float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    metrics_to_keep = (
        "q95_brier",
        "picp80",
        "crps_mase_scaled",
        "interval_score80_mase_scaled",
        "width80_mase_scaled",
    )
    for model in (MODEL_CANDIDATE, *COMPARATORS):
        root = candidate_root if model == MODEL_CANDIDATE else baseline_root
        for seed in seeds:
            clean = clean_factors(root, model, seed)
            for duration in DURATIONS:
                prediction = load_trailing_prediction(
                    prediction_path(root, model, seed), duration
                )
                duration_factors = fit_multiplicative_sigma_calibration(
                    prediction, bounds
                )
                methods = {
                    "uncalibrated": (prediction, np.ones_like(clean)),
                    "locked_clean_calibration": (
                        apply_pollutant_scale(prediction, clean),
                        clean,
                    ),
                    "duration_specific_in_sample_diagnostic": (
                        apply_pollutant_scale(prediction, duration_factors),
                        duration_factors,
                    ),
                }
                for method, (calibrated, factors) in methods.items():
                    summary = outage_metrics(
                        calibrated,
                        data,
                        calibrated["horizons"].astype(int).tolist(),
                    )["summary"]
                    row: dict[str, object] = {
                        "model": model,
                        "seed": seed,
                        "duration_hours": duration,
                        "method": method,
                        "factors": json.dumps([float(value) for value in factors]),
                    }
                    row.update({metric: summary[metric] for metric in metrics_to_keep})
                    rows.append(row)
            log(f"calibration diagnostic complete: {model}, seed {seed}")
    detail = pd.DataFrame(rows)
    grouped = detail.groupby(["model", "duration_hours", "method"], sort=False)
    summary_rows: list[dict[str, object]] = []
    for keys, frame in grouped:
        row = {
            "model": keys[0],
            "duration_hours": keys[1],
            "method": keys[2],
            "seeds": len(frame),
        }
        for metric in metrics_to_keep:
            row[f"{metric}_mean"] = float(frame[metric].mean())
            row[f"{metric}_sd"] = float(frame[metric].std(ddof=1))
        summary_rows.append(row)
    return detail, pd.DataFrame(summary_rows)


def decision_record(
    primary: pd.DataFrame,
    per_seed: pd.DataFrame,
    calibration: pd.DataFrame,
) -> dict[str, object]:
    fixed = primary[primary["comparator"] == "impute_then_local_tcn"]
    fixed_seed = per_seed[per_seed["comparator"] == "impute_then_local_tcn"]
    fixed_all_positive = bool((fixed["relative_benefit_percent"] > 0).all())
    fixed_all_seed_wins = bool(fixed_seed["candidate_win"].all())
    fixed_all_ci = bool(fixed["candidate_better_ci95"].all())

    candidate_calibration = calibration[calibration["model"] == MODEL_CANDIDATE]
    calibration_comparison: list[dict[str, object]] = []
    for duration in DURATIONS:
        duration_rows = candidate_calibration[
            candidate_calibration["duration_hours"] == duration
        ].set_index("method")
        locked = duration_rows.loc["locked_clean_calibration"]
        diagnostic = duration_rows.loc["duration_specific_in_sample_diagnostic"]
        calibration_comparison.append(
            {
                "duration_hours": duration,
                "locked_picp80": locked["picp80_mean"],
                "diagnostic_picp80": diagnostic["picp80_mean"],
                "locked_q95_brier": locked["q95_brier_mean"],
                "diagnostic_q95_brier": diagnostic["q95_brier_mean"],
                "locked_crps_mase_scaled": locked["crps_mase_scaled_mean"],
                "diagnostic_crps_mase_scaled": diagnostic[
                    "crps_mase_scaled_mean"
                ],
                "coverage_error_reduced": abs(diagnostic["picp80_mean"] - 0.8)
                < abs(locked["picp80_mean"] - 0.8),
            }
        )

    return {
        "created_utc": utc_now(),
        "development_only": True,
        "holdout_accessed": False,
        "selected_candidate": MODEL_CANDIDATE,
        "candidate_vs_frozen_imputation_baseline": {
            "positive_mean_benefit_all_four_primary_cells": fixed_all_positive,
            "all_5_seed_wins_in_all_four_primary_cells": fixed_all_seed_wins,
            "bootstrap_ci_excludes_zero_all_four_primary_cells": fixed_all_ci,
            "minimum_relative_benefit_percent": float(
                fixed["relative_benefit_percent"].min()
            ),
            "maximum_relative_benefit_percent": float(
                fixed["relative_benefit_percent"].max()
            ),
        },
        "calibration_diagnostic": {
            "status": "post_selection_in_sample_development_diagnostic_only",
            "comparisons": calibration_comparison,
            "protocol_action": (
                "Retain the locked clean-validation calibration for the sealed holdout. "
                "Do not adopt duration-specific factors from this in-sample diagnostic "
                "without declaring a protocol amendment and obtaining new untouched data."
            ),
        },
        "freeze_recommendation": (
            "freeze_candidate_then_open_holdout_once"
            if fixed_all_positive and fixed_all_seed_wins
            else "do_not_freeze_candidate"
        ),
        "claim_boundary": (
            "The adaptive graph is a conservative residual refinement. Development gains "
            "over its frozen imputation base are consistent but small; report effect sizes "
            "and uncertainty and do not claim a large improvement."
        ),
    }


def artifact_manifest(
    output: Path,
    input_files: list[Path],
    output_files: list[Path],
) -> dict[str, object]:
    return {
        "created_utc": utc_now(),
        "development_only": True,
        "holdout_accessed": False,
        "input_sha256": {
            str(path.relative_to(ROOT)): sha256_file(path) for path in sorted(set(input_files))
        },
        "output_sha256": {
            str(path.relative_to(output)): sha256_file(path) for path in output_files
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, default=DEFAULT_CANDIDATE_ROOT)
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE_ROOT)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidate_root = args.candidate_root.resolve()
    baseline_root = args.baseline_root.resolve()
    output = (
        args.output.resolve()
        if args.output
        else candidate_root / "analysis"
    )
    launch = read_json(candidate_root / "launch_manifest.json")
    protocol = launch["signature"]["protocol"]
    if launch.get("holdout_accessed") is not False:
        raise RuntimeError("Candidate manifest does not preserve the sealed-holdout invariant")
    if read_json(candidate_root / "status.json").get("state") != "complete":
        raise RuntimeError("Candidate run is not complete")
    seeds = [int(seed) for seed in protocol["seeds"]]
    if seeds != [42, 123, 2026, 3407, 7777]:
        raise ValueError(f"Unexpected seed set: {seeds}")
    prepared = ROOT / protocol["prepared_development"]
    if sha256_file(prepared) != launch["signature"]["prepared_sha256"]:
        raise RuntimeError("Prepared development data hash differs from the training manifest")
    data = load_outage_data(prepared, development_only=True)
    statistics = protocol["statistics"]
    block_length = int(statistics["temporal_block_length_hours"])
    resamples = int(statistics["bootstrap_resamples"])
    bounds = tuple(float(value) for value in protocol["calibration"]["factor_bounds"])

    log("starting paired primary tests; sealed holdout remains unopened")
    primary, per_seed = paired_primary_tests(
        candidate_root,
        baseline_root,
        data,
        seeds,
        block_length,
        resamples,
    )
    log("starting calibration diagnostics")
    calibration_detail, calibration_summary = calibration_diagnostics(
        candidate_root,
        baseline_root,
        data,
        seeds,
        bounds,
    )
    decision = decision_record(primary, per_seed, calibration_summary)

    outputs = {
        "primary_paired_tests.csv": primary,
        "per_seed_effects.csv": per_seed,
        "calibration_diagnostics_per_seed.csv": calibration_detail,
        "calibration_diagnostics_summary.csv": calibration_summary,
    }
    output.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []
    for name, frame in outputs.items():
        path = output / name
        atomic_csv(frame, path)
        output_paths.append(path)
    decision_path = output / "development_decision.json"
    write_json(decision_path, decision)
    output_paths.append(decision_path)

    input_files = [
        Path(__file__).resolve(),
        ROOT / "src/aqriskformer/development_analysis.py",
        ROOT / "src/aqriskformer/statistics.py",
        ROOT / "src/aqriskformer/outage_evaluation.py",
        ROOT / "journal_protocol/comparison_protocol.json",
        candidate_root / "launch_manifest.json",
        candidate_root / "status.json",
        prepared,
    ]
    for model in (MODEL_CANDIDATE, *COMPARATORS):
        root = candidate_root if model == MODEL_CANDIDATE else baseline_root
        for seed in seeds:
            input_files.extend(
                [
                    prediction_path(root, model, seed),
                    run_dir(root, model, seed) / "calibration.npz",
                    run_dir(root, model, seed) / "complete.json",
                ]
            )
    manifest = artifact_manifest(output, input_files, output_paths)
    manifest_path = output / "analysis_manifest.json"
    write_json(manifest_path, manifest)
    log(f"analysis complete: {output}")
    log(f"freeze recommendation: {decision['freeze_recommendation']}")


if __name__ == "__main__":
    main()
