"""Generate the prespecified statistical report from the frozen 2025 run."""

from __future__ import annotations

import argparse
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
from aqriskformer.epa_aqs_holdout import (
    load_holdout_outage_data,
    verify_execution_freeze,
)
from aqriskformer.statistics import (
    diebold_mariano,
    holm_adjust,
    paired_moving_block_bootstrap,
)
from aqriskformer.utils import read_json, sha256_file, utc_now, write_json

PROTOCOL_PATH = ROOT / "journal_protocol/final_holdout_protocol.json"
FREEZE_PATH = ROOT / "journal_protocol/final_holdout_execution_freeze.json"


def log(message: str) -> None:
    print(f"[{utc_now()}] {message}", flush=True)


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        frame.to_csv(temporary, index=False)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def result_dir(root: Path, model: str, seed: int | None) -> Path:
    base = root / "runs" / model
    return base if seed is None else base / f"seed_{seed}"


def model_seed(final: dict[str, object], model: str, seed: int) -> int | None:
    return None if model in final["deterministic_comparators"] else seed


def calibrated_prediction(
    root: Path,
    model: str,
    seed: int | None,
    duration: int,
) -> dict[str, np.ndarray]:
    directory = result_dir(root, model, seed)
    prediction = load_trailing_prediction(directory / "trailing_test_predictions.npz", duration)
    with np.load(directory / "frozen_calibration.npz", allow_pickle=False) as archive:
        if set(archive.files) != {"factors"}:
            raise ValueError(f"Unexpected calibration schema: {directory}")
        factors = archive["factors"].copy()
    return apply_pollutant_scale(prediction, factors)


def primary_tests(
    root: Path,
    final: dict[str, object],
    data,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidate = final["candidate"]
    comparators = [
        *final["learned_comparators"],
        *final["deterministic_comparators"],
    ]
    seeds = [int(seed) for seed in final["seeds"]]
    durations = (6, 24)
    endpoints = tuple(final["endpoints"]["primary"])
    block = int(final["statistics"]["temporal_block_length_hours"])
    resamples = int(final["statistics"]["bootstrap_resamples"])
    rows: list[dict[str, object]] = []
    seed_rows: list[dict[str, object]] = []
    for comparator_index, comparator in enumerate(comparators):
        for duration in durations:
            candidate_by_endpoint = {endpoint: [] for endpoint in endpoints}
            comparator_by_endpoint = {endpoint: [] for endpoint in endpoints}
            for seed in seeds:
                candidate_prediction = calibrated_prediction(root, candidate, seed, duration)
                comparator_prediction = calibrated_prediction(
                    root,
                    comparator,
                    model_seed(final, comparator, seed),
                    duration,
                )
                verify_paired_predictions(candidate_prediction, comparator_prediction)
                candidate_losses = origin_macro_losses(candidate_prediction, data)
                comparator_losses = origin_macro_losses(comparator_prediction, data)
                for endpoint in endpoints:
                    candidate_grid = hourly_loss_grid(
                        candidate_prediction["origins"], candidate_losses[endpoint]
                    )
                    comparator_grid = hourly_loss_grid(
                        comparator_prediction["origins"], comparator_losses[endpoint]
                    )
                    candidate_by_endpoint[endpoint].append(candidate_grid)
                    comparator_by_endpoint[endpoint].append(comparator_grid)
                    valid = np.isfinite(candidate_grid) & np.isfinite(comparator_grid)
                    difference = candidate_grid[valid] - comparator_grid[valid]
                    comparator_mean = float(comparator_grid[valid].mean())
                    seed_rows.append(
                        {
                            "candidate": candidate,
                            "comparator": comparator,
                            "duration_hours": duration,
                            "endpoint": endpoint,
                            "seed": seed,
                            "paired_origins": int(valid.sum()),
                            "candidate_mean": float(candidate_grid[valid].mean()),
                            "comparator_mean": comparator_mean,
                            "mean_difference_candidate_minus_comparator": float(difference.mean()),
                            "relative_benefit_percent": float(
                                -100 * difference.mean() / comparator_mean
                            ),
                            "candidate_win": bool(difference.mean() < 0),
                        }
                    )
            for endpoint_index, endpoint in enumerate(endpoints):
                candidate_mean = finite_column_mean(candidate_by_endpoint[endpoint])
                comparator_mean = finite_column_mean(comparator_by_endpoint[endpoint])
                bootstrap = paired_moving_block_bootstrap(
                    candidate_mean,
                    comparator_mean,
                    block_length=block,
                    resamples=resamples,
                    seed=2025 + 100 * comparator_index + 10 * duration + endpoint_index,
                )
                dm = diebold_mariano(
                    candidate_mean,
                    comparator_mean,
                    horizon=1,
                    newey_west_lag=block - 1,
                )
                valid = np.isfinite(candidate_mean) & np.isfinite(comparator_mean)
                reference = float(comparator_mean[valid].mean())
                rows.append(
                    {
                        "candidate": candidate,
                        "comparator": comparator,
                        "condition": f"each_station_trailing_{duration}h_outage",
                        "duration_hours": duration,
                        "endpoint": endpoint,
                        "seeds_averaged_per_origin": len(seeds),
                        "paired_origins": int(valid.sum()),
                        "candidate_mean": float(candidate_mean[valid].mean()),
                        "comparator_mean": reference,
                        "mean_difference_candidate_minus_comparator": bootstrap["mean_difference"],
                        "relative_benefit_percent": float(
                            -100 * bootstrap["mean_difference"] / reference
                        ),
                        "bootstrap_ci95_lower": bootstrap["ci95_lower"],
                        "bootstrap_ci95_upper": bootstrap["ci95_upper"],
                        "candidate_better_ci95": bool(bootstrap["ci95_upper"] < 0),
                        "dm_statistic": dm["statistic"],
                        "dm_p_value": dm["p_value"],
                    }
                )
                log(f"primary report complete: {comparator}, {duration} h, {endpoint}")
    primary = pd.DataFrame(rows)
    primary["dm_p_holm"] = holm_adjust(primary["dm_p_value"].to_numpy())
    primary["dm_significant_holm_0_05"] = primary["dm_p_holm"] < 0.05
    return primary, pd.DataFrame(seed_rows)


def mean_metric_summaries(items: list[dict[str, object]]) -> dict[str, float | int | None]:
    if not items:
        return {}
    keys = set.intersection(*(set(item) for item in items))
    result: dict[str, float | int | None] = {}
    for key in sorted(keys):
        values = [item[key] for item in items if item[key] is not None]
        if not values or isinstance(values[0], str):
            continue
        if key in ("observed_targets", "valid_metric_cells"):
            result[key] = int(sum(int(value) for value in values))
        else:
            result[key] = float(np.mean(values))
    return result


def metric_summary_table(root: Path, final: dict[str, object]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    models = [
        final["candidate"],
        *final["learned_comparators"],
        *final["deterministic_comparators"],
    ]
    seeds = [int(seed) for seed in final["seeds"]]
    calibration_names = (
        "uncalibrated",
        "frozen_clean_validation_calibration",
    )
    for model in models:
        model_seeds: list[int | None] = (
            [None] if model in final["deterministic_comparators"] else seeds
        )
        for seed in model_seeds:
            metrics = read_json(result_dir(root, model, seed) / "test_metrics.json")
            condition_groups = {
                "clean": ["clean"],
                "random_block_6h": ["random_block_6h"],
                "random_block_24h": ["random_block_24h"],
                "each_station_full_history_dropout": [
                    f"station_full_{station}" for station in range(7)
                ],
                "each_station_trailing_6h_outage": [
                    f"station_trailing_6h_{station}" for station in range(7)
                ],
                "each_station_trailing_24h_outage": [
                    f"station_trailing_24h_{station}" for station in range(7)
                ],
            }
            for condition, keys in condition_groups.items():
                for calibration in calibration_names:
                    summaries = [metrics[key][calibration]["summary"] for key in keys]
                    summary = mean_metric_summaries(summaries)
                    rows.append(
                        {
                            "model": model,
                            "seed": seed,
                            "condition": condition,
                            "calibration": calibration,
                            **summary,
                        }
                    )
            natural = metrics["natural_comissingness"]
            for hours in (6, 24):
                station_records = natural[f"trailing_{hours}h"].values()
                for calibration in calibration_names:
                    summaries = [
                        item[calibration]["summary"]
                        for item in station_records
                        if item.get("forecast_origins", 0) > 0
                    ]
                    if summaries:
                        summary = mean_metric_summaries(summaries)
                        events = sum(
                            int(item["forecast_origins"])
                            for item in natural[f"trailing_{hours}h"].values()
                        )
                        rows.append(
                            {
                                "model": model,
                                "seed": seed,
                                "condition": f"natural_comissingness_{hours}h",
                                "calibration": calibration,
                                "natural_forecast_origins": events,
                                **summary,
                            }
                        )
    return pd.DataFrame(rows)


def clean_guardrail(summary: pd.DataFrame, final: dict[str, object]) -> pd.DataFrame:
    calibrated = summary[
        (summary["condition"] == "clean")
        & (summary["calibration"] == "frozen_clean_validation_calibration")
    ]
    candidate = calibrated[calibrated["model"] == final["candidate"]]
    limit = float(
        final["endpoints"]["clean_guardrail"]["maximum_relative_degradation_vs_each_local_control"]
    )
    rows = []
    for control in final["endpoints"]["clean_guardrail"]["local_controls"]:
        baseline = calibrated[calibrated["model"] == control]
        for endpoint in final["endpoints"]["clean_guardrail"]["endpoints"]:
            candidate_mean = float(candidate[endpoint].mean())
            control_mean = float(baseline[endpoint].mean())
            degradation = (candidate_mean - control_mean) / control_mean
            rows.append(
                {
                    "candidate": final["candidate"],
                    "control": control,
                    "endpoint": endpoint,
                    "candidate_mean": candidate_mean,
                    "control_mean": control_mean,
                    "relative_degradation": degradation,
                    "maximum_allowed": limit,
                    "guardrail_pass": bool(degradation <= limit),
                }
            )
    return pd.DataFrame(rows)


def verify_run_complete(root: Path, final: dict[str, object]) -> list[Path]:
    status = read_json(root / "status.json")
    if status.get("state") != "complete" or status.get("test_labels_used_for_fitting") is not False:
        raise RuntimeError("Frozen holdout run is not complete")
    files = [root / "status.json", root / "test_access_log.json"]
    models = [final["candidate"], *final["learned_comparators"]]
    for model in models:
        for seed in final["seeds"]:
            directory = result_dir(root, model, int(seed))
            marker = read_json(directory / "complete.json")
            for name, expected in marker["artifact_hashes"].items():
                path = directory / name
                if sha256_file(path) != expected:
                    raise RuntimeError(f"Holdout artifact hash mismatch: {path}")
                files.append(path)
            files.append(directory / "complete.json")
    for model in final["deterministic_comparators"]:
        directory = result_dir(root, model, None)
        marker = read_json(directory / "complete.json")
        for name, expected in marker["artifact_hashes"].items():
            path = directory / name
            if sha256_file(path) != expected:
                raise RuntimeError(f"Holdout artifact hash mismatch: {path}")
            files.append(path)
        files.append(directory / "complete.json")
    return files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    verify_execution_freeze(ROOT, FREEZE_PATH)
    final = read_json(PROTOCOL_PATH)
    run_root = project_path(final["output"])
    inputs = verify_run_complete(run_root, final)
    prepared = project_path(final["data"]["prepared_holdout"])
    data = load_holdout_outage_data(prepared)
    output = args.output.resolve() if args.output else run_root / "analysis"

    primary, per_seed = primary_tests(run_root, final, data)
    summary = metric_summary_table(run_root, final)
    guardrail = clean_guardrail(summary, final)
    outputs = {
        "primary_paired_tests.csv": primary,
        "per_seed_primary_effects.csv": per_seed,
        "test_metrics_summary.csv": summary,
        "clean_guardrail.csv": guardrail,
    }
    output.mkdir(parents=True, exist_ok=True)
    output_paths = []
    for name, frame in outputs.items():
        path = output / name
        atomic_csv(frame, path)
        output_paths.append(path)
    decision = {
        "created_utc": utc_now(),
        "holdout_accessed": True,
        "test_labels_used_for_fitting": False,
        "candidate": final["candidate"],
        "primary_comparisons": len(primary),
        "candidate_mean_better": int(
            (primary["mean_difference_candidate_minus_comparator"] < 0).sum()
        ),
        "candidate_better_bootstrap_ci95": int(primary["candidate_better_ci95"].sum()),
        "candidate_better_holm_dm_0_05": int(
            (
                (primary["mean_difference_candidate_minus_comparator"] < 0)
                & primary["dm_significant_holm_0_05"]
            ).sum()
        ),
        "clean_guardrails_passed": int(guardrail["guardrail_pass"].sum()),
        "clean_guardrails_total": len(guardrail),
        "reporting_rule": (
            "Report every primary row, including unfavorable or nonsignificant results. "
            "The holdout is confirmatory and cannot be used for another model revision."
        ),
    }
    decision_path = output / "holdout_decision.json"
    write_json(decision_path, decision)
    output_paths.append(decision_path)
    source_inputs = [
        Path(__file__).resolve(),
        PROTOCOL_PATH,
        FREEZE_PATH,
        prepared,
        *inputs,
    ]
    write_json(
        output / "analysis_manifest.json",
        {
            "created_utc": utc_now(),
            "holdout_accessed": True,
            "test_labels_used_for_fitting": False,
            "input_sha256": {
                str(path.relative_to(ROOT)): sha256_file(path)
                for path in sorted(set(source_inputs))
            },
            "output_sha256": {path.name: sha256_file(path) for path in output_paths},
        },
    )
    log(f"frozen holdout report complete: {output}")


if __name__ == "__main__":
    main()
