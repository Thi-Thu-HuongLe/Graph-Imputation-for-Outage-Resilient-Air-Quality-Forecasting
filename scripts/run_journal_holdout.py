"""Execute the single frozen 2025 confirmation without fitting on test labels."""

from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_journal_development import (
    evaluation_conditions,
    exclusive_lock,
    make_loader,
    make_model,
    predict_deterministic,
    predict_learned,
)

from aqriskformer.epa_aqs_holdout import (
    load_holdout_outage_data,
    prepare_epa_holdout,
    verify_execution_freeze,
)
from aqriskformer.outage_data import (
    OutageWindows,
    natural_comissingness_origins,
)
from aqriskformer.outage_evaluation import apply_sigma_calibration, outage_metrics
from aqriskformer.utils import (
    environment_snapshot,
    read_json,
    sha256_file,
    utc_now,
    write_json,
)

PROTOCOL_PATH = ROOT / "journal_protocol/final_holdout_protocol.json"
FREEZE_PATH = ROOT / "journal_protocol/final_holdout_execution_freeze.json"
PREPROCESSING_PATH = ROOT / "journal_protocol/development_preprocessing_protocol.json"


def log(message: str) -> None:
    print(f"[{utc_now()}] {message}", flush=True)


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def runtime_protocol(final: dict[str, object]) -> dict[str, object]:
    protocol = read_json(ROOT / "journal_protocol/comparison_protocol.json")
    forecast = final["forecast"]
    for name in (
        "lookback",
        "horizon",
        "reported_horizons",
        "test_stride",
        "fill_limit",
        "corruption_seed",
    ):
        protocol[name] = forecast[name]
    return protocol


def select_device(name: str) -> torch.device:
    device = torch.device(name)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
        index = device.index or 0
        if index >= torch.cuda.device_count():
            raise RuntimeError(f"CUDA device {index} does not exist")
    return device


def calibration_factors(directory: Path, pollutants: int) -> np.ndarray:
    path = directory / "calibration.npz"
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"factors"}:
            raise ValueError(f"Unexpected frozen calibration schema: {path}")
        factors = archive["factors"].copy()
    if factors.shape != (pollutants,) or not np.isfinite(factors).all() or not (factors > 0).all():
        raise ValueError(f"Invalid frozen calibration factors: {path}")
    return factors


def evaluate_with_frozen_calibration(
    predictor,
    data,
    protocol: dict[str, object],
    calibration: np.ndarray,
) -> tuple[dict[str, object], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Evaluate all locked conditions; never estimate calibration from test targets."""
    metrics: dict[str, object] = {}
    trailing_parts = {
        hours: {key: [] for key in ("mu", "sigma", "target", "mask")} for hours in (6, 24)
    }
    clean = predictor("clean")
    for condition, station in evaluation_conditions(len(data.stations), smoke=False):
        log(f"TEST condition={condition}")
        prediction = clean if condition == "clean" else predictor(condition)
        metrics[condition] = {
            "uncalibrated": outage_metrics(
                prediction,
                data,
                list(protocol["reported_horizons"]),
                station=station,
            ),
            "frozen_clean_validation_calibration": outage_metrics(
                apply_sigma_calibration(prediction, calibration),
                data,
                list(protocol["reported_horizons"]),
                station=station,
            ),
        }
        if station is not None and condition.startswith("station_trailing_"):
            hours = int(condition.split("_")[2].removesuffix("h"))
            for key in trailing_parts[hours]:
                trailing_parts[hours][key].append(prediction[key][:, :, station, :])
    bundle: dict[str, np.ndarray] = {
        "origins": clean["origin"],
        "horizons": np.asarray(protocol["reported_horizons"], dtype=np.int16),
    }
    for hours, parts in trailing_parts.items():
        for key, values in parts.items():
            bundle[f"h{hours}_{key}"] = np.stack(values, axis=2)
    return metrics, clean, bundle


def evaluate_natural_comissingness_test(
    predict_windows,
    data,
    protocol: dict[str, object],
    calibration: np.ndarray,
) -> dict[str, object]:
    hourly_protocol = {**protocol, "test_stride": 1}
    hourly = OutageWindows(data, hourly_protocol, "test")
    output: dict[str, object] = {}
    for hours in (6, 24):
        selected = natural_comissingness_origins(data, hourly.origins, hours)
        station_results: dict[str, object] = {}
        for station, station_name in enumerate(data.stations):
            positions = np.flatnonzero(selected[:, station])
            if not len(positions):
                station_results[station_name] = {"forecast_origins": 0, "metrics": None}
                continue
            windows = copy.copy(hourly)
            windows.origins = hourly.origins[positions]
            prediction = predict_windows(windows)
            station_results[station_name] = {
                "forecast_origins": len(positions),
                "origin_timestamps_utc": [
                    str(data.timestamps[origin]) for origin in windows.origins
                ],
                "uncalibrated": outage_metrics(
                    prediction,
                    data,
                    list(protocol["reported_horizons"]),
                    station=station,
                ),
                "frozen_clean_validation_calibration": outage_metrics(
                    apply_sigma_calibration(prediction, calibration),
                    data,
                    list(protocol["reported_horizons"]),
                    station=station,
                ),
            }
        output[f"trailing_{hours}h"] = station_results
    return output


def verify_complete(directory: Path) -> bool:
    marker_path = directory / "complete.json"
    if not marker_path.is_file():
        return False
    marker = read_json(marker_path)
    if marker.get("state") != "complete" or marker.get("test_labels_used_for_fitting") is not False:
        raise RuntimeError(f"Invalid completed holdout run: {directory}")
    for name, expected in marker["artifact_hashes"].items():
        if sha256_file(directory / name) != expected:
            raise RuntimeError(f"Completed holdout artifact changed: {directory / name}")
    return True


def save_test_artifacts(
    directory: Path,
    model_name: str,
    seed: int | None,
    metrics: dict[str, object],
    clean: dict[str, np.ndarray],
    trailing: dict[str, np.ndarray],
    calibration: np.ndarray,
    seconds: float,
    parameters: int,
    peak_gpu_memory: int | None,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "test_metrics.json": directory / "test_metrics.json",
        "clean_test_predictions.npz": directory / "clean_test_predictions.npz",
        "trailing_test_predictions.npz": directory / "trailing_test_predictions.npz",
        "frozen_calibration.npz": directory / "frozen_calibration.npz",
    }
    write_json(paths["test_metrics.json"], metrics)
    np.savez_compressed(paths["clean_test_predictions.npz"], **clean)
    np.savez_compressed(paths["trailing_test_predictions.npz"], **trailing)
    np.savez_compressed(paths["frozen_calibration.npz"], factors=calibration)
    write_json(
        directory / "complete.json",
        {
            "state": "complete",
            "model": model_name,
            "seed": seed,
            "test_only": True,
            "test_labels_used_for_fitting": False,
            "inference_and_evaluation_seconds": seconds,
            "parameters": parameters,
            "peak_gpu_memory_bytes": peak_gpu_memory,
            "artifact_hashes": {name: sha256_file(path) for name, path in paths.items()},
        },
    )


def prepare_once(final: dict[str, object], freeze: dict[str, object]) -> Path:
    data_protocol = final["data"]
    prepared_path = project_path(data_protocol["prepared_holdout"])
    report_path = project_path(data_protocol["preparation_report"])
    availability_path = project_path(data_protocol["availability_table"])
    if prepared_path.is_file() or report_path.is_file():
        if not (prepared_path.is_file() and report_path.is_file()):
            raise RuntimeError("Partial holdout preparation artifacts require manual audit")
        report = read_json(report_path)
        if report.get("status") != "PASS_HOLDOUT_PREPARED_AFTER_FINAL_FREEZE":
            raise RuntimeError("Existing holdout preparation report is invalid")
        if sha256_file(prepared_path) != report["prepared_sha256"]:
            raise RuntimeError("Prepared holdout hash differs from its report")
        return prepared_path

    output_root = project_path(final["output"])
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(
        output_root / "test_access_log.json",
        {
            "access_started_utc": utc_now(),
            "execution_freeze_sha256": sha256_file(FREEZE_PATH),
            "protocol_sha256": sha256_file(PROTOCOL_PATH),
            "holdout_manifest_sha256": final["data"]["sealed_manifest_sha256"],
            "freeze_authorized": freeze["holdout_evaluation_authorized"],
            "purpose": "single frozen 2025 confirmation",
        },
    )
    log("FINAL FREEZE VERIFIED; opening sealed 2025 sources for the first confirmation")
    prepared, report, availability = prepare_epa_holdout(
        PROTOCOL_PATH,
        PREPROCESSING_PATH,
        project_path(data_protocol["sealed_raw_root"]),
        project_path(data_protocol["development_prepared"]),
    )
    prepared.save(prepared_path)
    availability_path.parent.mkdir(parents=True, exist_ok=True)
    availability.to_csv(availability_path, index=False)
    report.update(
        completed_utc=utc_now(),
        execution_freeze_sha256=sha256_file(FREEZE_PATH),
        prepared_file=str(prepared_path.relative_to(ROOT)),
        prepared_sha256=sha256_file(prepared_path),
        availability_file=str(availability_path.relative_to(ROOT)),
        availability_sha256=sha256_file(availability_path),
    )
    write_json(report_path, report)
    return prepared_path


def learned_source(final: dict[str, object], model: str, seed: int) -> Path:
    key = "candidate" if model == final["candidate"] else "comparators"
    return project_path(final["checkpoint_roots"][key]) / model / f"seed_{seed}"


def run_learned_test(
    final: dict[str, object],
    protocol: dict[str, object],
    data,
    model_name: str,
    seed: int,
    device: torch.device,
    output: Path,
) -> None:
    if verify_complete(output):
        log(f"SKIP verified {model_name}/seed_{seed}")
        return
    source = learned_source(final, model_name, seed)
    checkpoint = torch.load(source / "best.pt", map_location=device, weights_only=False)
    if checkpoint.get("model_name") != model_name or int(checkpoint.get("seed", -1)) != seed:
        raise RuntimeError(f"Frozen checkpoint identity mismatch: {source}")
    model = make_model(checkpoint["protocol"], data, model_name, seed, device)
    model.load_state_dict(checkpoint["model"], strict=True)
    calibration = calibration_factors(source, data.n_pollutants)
    windows = OutageWindows(data, protocol, "test")
    loader = make_loader(windows, protocol, False, seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    metrics, clean, trailing = evaluate_with_frozen_calibration(
        lambda condition: predict_learned(model, loader, condition, protocol, device),
        data,
        protocol,
        calibration,
    )
    metrics["natural_comissingness"] = evaluate_natural_comissingness_test(
        lambda selected: predict_learned(
            model,
            make_loader(selected, protocol, False, seed),
            "clean",
            protocol,
            device,
        ),
        data,
        protocol,
        calibration,
    )
    save_test_artifacts(
        output,
        model_name,
        seed,
        metrics,
        clean,
        trailing,
        calibration,
        time.perf_counter() - started,
        sum(parameter.numel() for parameter in model.parameters()),
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
    )


def run_deterministic_test(
    final: dict[str, object],
    protocol: dict[str, object],
    data,
    model_name: str,
    device: torch.device,
    output: Path,
) -> None:
    if verify_complete(output):
        log(f"SKIP verified {model_name}")
        return
    source = project_path(final["checkpoint_roots"]["comparators"]) / model_name
    with np.load(source / "training_residual_sigma.npz", allow_pickle=False) as archive:
        sigma = torch.from_numpy(archive["sigma"].copy()).to(device)
    calibration = calibration_factors(source, data.n_pollutants)
    windows = OutageWindows(data, protocol, "test")
    loader = make_loader(windows, protocol, False, 0)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    metrics, clean, trailing = evaluate_with_frozen_calibration(
        lambda condition: predict_deterministic(
            model_name, sigma, loader, condition, protocol, device
        ),
        data,
        protocol,
        calibration,
    )
    metrics["natural_comissingness"] = evaluate_natural_comissingness_test(
        lambda selected: predict_deterministic(
            model_name,
            sigma,
            make_loader(selected, protocol, False, 0),
            "clean",
            protocol,
            device,
        ),
        data,
        protocol,
        calibration,
    )
    save_test_artifacts(
        output,
        model_name,
        None,
        metrics,
        clean,
        trailing,
        calibration,
        time.perf_counter() - started,
        0,
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "run", "all"), default="all")
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    freeze = verify_execution_freeze(ROOT, FREEZE_PATH)
    final = read_json(PROTOCOL_PATH)
    if (
        sha256_file(
            final["candidate_selection_freeze"]
            if Path(final["candidate_selection_freeze"]).is_absolute()
            else ROOT / final["candidate_selection_freeze"]
        )
        != final["candidate_selection_freeze_sha256"]
    ):
        raise RuntimeError("Candidate-selection freeze hash mismatch")
    output_root = project_path(final["output"])
    with exclusive_lock(output_root / "runner.lock"):
        prepared_path = prepare_once(final, freeze)
        if args.stage == "prepare":
            log(f"holdout preparation complete: {prepared_path}")
            return
        data = load_holdout_outage_data(prepared_path)
        protocol = runtime_protocol(final)
        device = select_device(args.device)
        learned = [final["candidate"], *final["learned_comparators"]]
        expected = len(learned) * len(final["seeds"]) + len(final["deterministic_comparators"])
        write_json(
            output_root / "status.json",
            {
                "state": "running",
                "started_utc": utc_now(),
                "expected_fits": expected,
                "device": str(device),
                "holdout_accessed": True,
                "test_labels_used_for_fitting": False,
            },
        )
        write_json(output_root / "environment.json", environment_snapshot())
        completed = 0
        for model in learned:
            for seed in final["seeds"]:
                log(f"EVALUATE {model}/seed_{seed} on frozen 2025 test")
                run_learned_test(
                    final,
                    protocol,
                    data,
                    model,
                    int(seed),
                    device,
                    output_root / "runs" / model / f"seed_{seed}",
                )
                completed += 1
                write_json(
                    output_root / "status.json",
                    {
                        "state": "running",
                        "updated_utc": utc_now(),
                        "expected_fits": expected,
                        "completed_fits": completed,
                        "current": f"{model}/seed_{seed}",
                        "device": str(device),
                        "holdout_accessed": True,
                        "test_labels_used_for_fitting": False,
                    },
                )
        for model in final["deterministic_comparators"]:
            log(f"EVALUATE {model} on frozen 2025 test")
            run_deterministic_test(
                final,
                protocol,
                data,
                model,
                device,
                output_root / "runs" / model,
            )
            completed += 1
        write_json(
            output_root / "status.json",
            {
                "state": "complete",
                "completed_utc": utc_now(),
                "expected_fits": expected,
                "completed_fits": completed,
                "device": str(device),
                "holdout_accessed": True,
                "test_labels_used_for_fitting": False,
            },
        )
        log(f"frozen holdout evaluation complete: {completed}/{expected}")


if __name__ == "__main__":
    main()
