"""Run the common train/validation protocol without opening the sealed 2025 data."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import shutil
import sys
import time
import traceback
from contextlib import contextmanager, nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aqriskformer.outage_data import (
    OutageWindows,
    causal_outage_inputs,
    load_outage_data,
    natural_comissingness_origins,
)
from aqriskformer.outage_evaluation import (
    apply_sigma_calibration,
    fit_multiplicative_sigma_calibration,
    matched_gaussian_loss,
    outage_metrics,
    scaled_thresholds,
)
from aqriskformer.outage_models import (
    LEARNED_MODELS,
    build_outage_model,
    deterministic_forecast,
)
from aqriskformer.utils import (
    environment_snapshot,
    read_json,
    seed_everything,
    sha256_file,
    utc_now,
    write_json,
)

PROTOCOL_PATH = ROOT / "journal_protocol/comparison_protocol.json"
DEFAULT_OUTPUT = ROOT / "experiment_protocol/results_journal/development"
CHECKPOINT_SCHEMA_VERSION = 1


def log(message: str) -> None:
    print(f"[{utc_now()}] {message}", flush=True)


@contextmanager
def exclusive_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        handle.seek(0, 2)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def make_loader(
    dataset: OutageWindows,
    protocol: dict[str, object],
    shuffle: bool,
    seed: int,
) -> DataLoader:
    training = protocol["training"]
    return DataLoader(
        dataset,
        batch_size=int(training["batch_size"]),
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def on_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def autocast_context(device: torch.device):
    return torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()


def make_model(protocol: dict[str, object], data, name: str, seed: int, device: torch.device):
    seed_everything(seed)
    architecture = protocol["architecture"]
    hidden_by_model = architecture.get("hidden_by_model", {})
    model = build_outage_model(
        name,
        features=data.n_features,
        pollutants=data.n_pollutants,
        stations=len(data.stations),
        horizon=int(protocol["horizon"]),
        graph=torch.from_numpy(data.graph),
        hidden=int(hidden_by_model.get(name, architecture["hidden"])),
        layers=int(architecture["layers"]),
        dropout=float(architecture["dropout"]),
    ).to(device)
    seed_everything(seed + 10000)
    return model


def initialize_impute_adapter(
    model: torch.nn.Module,
    checkpoint_path: Path,
    protocol: dict[str, object],
    seed: int,
    device: torch.device,
) -> str:
    """Load and freeze the shared local path from a verified imputation baseline."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("model_name") != "impute_then_local_tcn":
        raise ValueError(f"Adapter base is not impute_then_local_tcn: {checkpoint_path}")
    if int(checkpoint.get("seed", -1)) != seed:
        raise ValueError(f"Adapter base seed does not match seed {seed}: {checkpoint_path}")
    base_protocol = checkpoint.get("protocol", {})
    compatibility_fields = (
        "prepared_development",
        "lookback",
        "horizon",
        "reported_horizons",
        "architecture",
        "training",
        "fill_limit",
    )
    for field in compatibility_fields:
        if base_protocol.get(field) != protocol.get(field):
            raise ValueError(f"Adapter base protocol differs in '{field}': {checkpoint_path}")
    model_state = model.state_dict()
    base_state = checkpoint["model"]
    required_prefixes = ("encoder.", "local_head.")
    required = {
        key for key in model_state if key.startswith(required_prefixes)
    } | {"graph"}
    if not required.issubset(base_state):
        missing = sorted(required - set(base_state))
        raise ValueError(f"Adapter base checkpoint lacks shared state: {missing}")
    for key in required:
        if model_state[key].shape != base_state[key].shape:
            raise ValueError(f"Adapter base tensor shape differs for '{key}'")
        model_state[key] = base_state[key]
    model.load_state_dict(model_state, strict=True)
    freeze_base = getattr(model, "freeze_base", None)
    if freeze_base is None:
        raise TypeError("The selected adapter model does not expose freeze_base()")
    freeze_base()
    return sha256_file(checkpoint_path)


def atomic_torch_save(value: object, path: Path) -> None:
    """Write a Torch artifact atomically so an interrupted write cannot corrupt it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    protocol: dict[str, object],
    model_name: str,
    seed: int,
    epoch: int,
    score: float,
) -> None:
    atomic_torch_save(
        {
            "model": model.state_dict(),
            "protocol": protocol,
            "model_name": model_name,
            "seed": seed,
            "epoch": epoch,
            "selection_score": score,
        },
        path,
    )


def save_training_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    protocol: dict[str, object],
    training: dict[str, object],
    model_name: str,
    seed: int,
    epoch: int,
    best: float,
    best_epoch: int,
    stale: int,
    history: list[dict[str, object]],
    elapsed_training_seconds: float,
    peak_gpu_memory_bytes: int | None,
    device: torch.device,
) -> None:
    """Save all state required to continue at the next epoch exactly."""
    loader_generator = getattr(train_loader, "generator", None)
    checkpoint: dict[str, object] = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "protocol": protocol,
        "training": training,
        "model_name": model_name,
        "seed": seed,
        "epoch": epoch,
        "best_selection_score": best,
        "best_epoch": best_epoch,
        "stale_epochs": stale,
        "history": history,
        "elapsed_training_seconds": elapsed_training_seconds,
        "peak_gpu_memory_bytes": peak_gpu_memory_bytes,
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "loader_generator_state": (
            loader_generator.get_state() if loader_generator is not None else None
        ),
    }
    if device.type == "cuda":
        checkpoint["cuda_rng_state"] = torch.cuda.get_rng_state(device).cpu()
    atomic_torch_save(checkpoint, path)


def load_training_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    protocol: dict[str, object],
    training: dict[str, object],
    model_name: str,
    seed: int,
    device: torch.device,
) -> dict[str, object] | None:
    """Restore an exact epoch-boundary checkpoint, or return None when absent."""
    if not path.is_file():
        return None
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    expected = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "protocol": protocol,
        "training": training,
        "model_name": model_name,
        "seed": seed,
    }
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise ValueError(f"Incompatible resume checkpoint field '{key}': {path}")
    history = checkpoint.get("history")
    epoch = int(checkpoint.get("epoch", 0))
    if not isinstance(history, list) or not history or int(history[-1]["epoch"]) != epoch:
        raise ValueError(f"Resume checkpoint has inconsistent epoch history: {path}")
    model.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    random.setstate(checkpoint["python_rng_state"])
    np.random.set_state(checkpoint["numpy_rng_state"])
    torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
    loader_generator = getattr(train_loader, "generator", None)
    loader_state = checkpoint.get("loader_generator_state")
    if loader_generator is not None and loader_state is not None:
        loader_generator.set_state(loader_state.cpu())
    if device.type == "cuda" and checkpoint.get("cuda_rng_state") is not None:
        torch.cuda.set_rng_state(checkpoint["cuda_rng_state"].cpu(), device)
    return checkpoint


def preserve_legacy_partial_run(directory: Path) -> Path | None:
    """Preserve a pre-resume partial run before a scientifically clean restart.

    Older checkpoints contain only the best model weights. AdamW moments and RNG
    states cannot be reconstructed, so silently treating them as exact resume
    checkpoints would change the locked training protocol.
    """
    if (directory / "last.pt").exists() or (directory / "complete.json").exists():
        return None
    artifacts = [path for path in (directory / "best.pt", directory / "history.json") if path.exists()]
    if not artifacts:
        return None
    history = read_json(directory / "history.json") if (directory / "history.json").exists() else []
    last_epoch = int(history[-1]["epoch"]) if history else 0
    snapshot = directory / "resume_snapshots" / f"legacy_partial_epoch_{last_epoch:03d}"
    snapshot.mkdir(parents=True, exist_ok=True)
    for source in artifacts:
        target = snapshot / source.name
        if not target.exists():
            shutil.copy2(source, target)
    manifest_path = snapshot / "snapshot.json"
    if not manifest_path.exists():
        write_json(
            manifest_path,
            {
                "created_utc": utc_now(),
                "reason": "legacy checkpoint lacked optimizer and RNG state; exact restart required",
                "last_recorded_epoch": last_epoch,
                "artifact_hashes": {
                    path.name: sha256_file(snapshot / path.name) for path in artifacts
                },
            },
        )
    return snapshot


@torch.inference_mode()
def predict_learned(
    model: torch.nn.Module,
    loader: DataLoader,
    condition: str,
    protocol: dict[str, object],
    device: torch.device,
) -> dict[str, np.ndarray]:
    model.eval()
    generator = torch.Generator(device=device).manual_seed(int(protocol["corruption_seed"]))
    indices = torch.tensor(protocol["reported_horizons"], device=device) - 1
    collected = {key: [] for key in ("mu", "sigma", "target", "mask", "origin")}
    for batch in loader:
        batch = causal_outage_inputs(
            on_device(batch, device),
            condition,
            generator,
            fill_limit=int(protocol["fill_limit"]),
        )
        with autocast_context(device):
            output = model(batch)
        for key in ("mu", "sigma"):
            collected[key].append(output[key].index_select(1, indices).float().cpu().numpy())
        collected["target"].append(batch["target"].index_select(1, indices).cpu().numpy())
        collected["mask"].append(batch["target_mask"].index_select(1, indices).cpu().numpy())
        collected["origin"].append(batch["origin"].cpu().numpy())
    return {key: np.concatenate(parts) for key, parts in collected.items()}


@torch.inference_mode()
def predict_deterministic(
    name: str,
    sigma: torch.Tensor,
    loader: DataLoader,
    condition: str,
    protocol: dict[str, object],
    device: torch.device,
) -> dict[str, np.ndarray]:
    generator = torch.Generator(device=device).manual_seed(int(protocol["corruption_seed"]))
    indices = torch.tensor(protocol["reported_horizons"], device=device) - 1
    collected = {key: [] for key in ("mu", "sigma", "target", "mask", "origin")}
    for batch in loader:
        batch = causal_outage_inputs(
            on_device(batch, device),
            condition,
            generator,
            fill_limit=int(protocol["fill_limit"]),
        )
        output = deterministic_forecast(name, batch, sigma)
        for key in ("mu", "sigma"):
            collected[key].append(output[key].index_select(1, indices).cpu().numpy())
        collected["target"].append(batch["target"].index_select(1, indices).cpu().numpy())
        collected["mask"].append(batch["target_mask"].index_select(1, indices).cpu().numpy())
        collected["origin"].append(batch["origin"].cpu().numpy())
    return {key: np.concatenate(parts) for key, parts in collected.items()}


@torch.inference_mode()
def fit_deterministic_sigma(
    name: str,
    loader: DataLoader,
    protocol: dict[str, object],
    data,
    device: torch.device,
) -> torch.Tensor:
    shape = (int(protocol["horizon"]), len(data.stations), data.n_pollutants)
    sum_squared = torch.zeros(shape, device=device, dtype=torch.float64)
    count = torch.zeros(shape, device=device, dtype=torch.float64)
    placeholder = torch.ones(shape, device=device)
    generator = torch.Generator(device=device).manual_seed(int(protocol["corruption_seed"]))
    for batch in loader:
        batch = causal_outage_inputs(
            on_device(batch, device),
            "clean",
            generator,
            fill_limit=int(protocol["fill_limit"]),
        )
        output = deterministic_forecast(name, batch, placeholder)
        valid = batch["target_mask"].bool()
        error_squared = (batch["target"] - output["mu"]).double().square()
        sum_squared += torch.where(valid, error_squared, 0).sum(0)
        count += valid.sum(0)
    variance = sum_squared / count.clamp_min(1)
    fallback = torch.nanmedian(torch.where(count > 0, variance.sqrt(), torch.nan))
    sigma = torch.where(count > 0, variance.sqrt(), fallback).clamp_min(1e-3)
    if not torch.isfinite(sigma).all():
        raise ValueError(f"Could not fit finite training residual scales for {name}")
    return sigma.float()


def evaluation_conditions(stations: int, smoke: bool) -> list[tuple[str, int | None]]:
    if smoke:
        return [("clean", None), ("station_trailing_6h_0", 0), ("station_trailing_24h_0", 0)]
    conditions: list[tuple[str, int | None]] = [
        ("clean", None),
        ("random_block_6h", None),
        ("random_block_24h", None),
    ]
    conditions.extend((f"station_full_{station}", station) for station in range(stations))
    for hours in (6, 24):
        conditions.extend(
            (f"station_trailing_{hours}h_{station}", station) for station in range(stations)
        )
    return conditions


def evaluate(
    predictor,
    data,
    protocol: dict[str, object],
    conditions: list[tuple[str, int | None]],
) -> tuple[dict[str, object], dict[str, np.ndarray], np.ndarray, dict[str, np.ndarray]]:
    metrics: dict[str, object] = {}
    trailing_parts = {
        hours: {key: [] for key in ("mu", "sigma", "target", "mask")} for hours in (6, 24)
    }
    clean_prediction = predictor("clean")
    bounds = tuple(float(value) for value in protocol["calibration"]["factor_bounds"])
    calibration = fit_multiplicative_sigma_calibration(clean_prediction, bounds)
    for condition, station in conditions:
        log(f"VALIDATION condition={condition}")
        prediction = clean_prediction if condition == "clean" else predictor(condition)
        metrics[condition] = {
            "uncalibrated": outage_metrics(
                prediction,
                data,
                list(protocol["reported_horizons"]),
                station=station,
            ),
            "calibrated": outage_metrics(
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
    trailing_bundle: dict[str, np.ndarray] = {
        "origins": clean_prediction["origin"],
        "horizons": np.asarray(protocol["reported_horizons"], dtype=np.int16),
    }
    for hours, parts in trailing_parts.items():
        if parts["mu"]:
            for key, values in parts.items():
                trailing_bundle[f"h{hours}_{key}"] = np.stack(values, axis=2)
    return metrics, clean_prediction, calibration, trailing_bundle


def evaluate_natural_comissingness(
    predictor,
    data,
    protocol: dict[str, object],
    calibration: np.ndarray,
) -> dict[str, object]:
    event_protocol = {**protocol, "validation_stride": 1}
    hourly_windows = OutageWindows(data, event_protocol, "validation")
    output = {}
    for hours in protocol["natural_comissingness"]["durations_hours"]:
        selected = natural_comissingness_origins(data, hourly_windows.origins, int(hours))
        station_results = {}
        for station, station_name in enumerate(data.stations):
            positions = np.flatnonzero(selected[:, station])
            if not len(positions):
                station_results[station_name] = {"forecast_origins": 0, "metrics": None}
                continue
            event_windows = copy.copy(hourly_windows)
            event_windows.origins = hourly_windows.origins[positions]
            prediction = predictor(event_windows)
            station_results[station_name] = {
                "forecast_origins": len(positions),
                "origin_timestamps_utc": [
                    str(data.timestamps[origin]) for origin in event_windows.origins
                ],
                "uncalibrated": outage_metrics(
                    prediction,
                    data,
                    list(protocol["reported_horizons"]),
                    station=station,
                ),
                "calibrated": outage_metrics(
                    apply_sigma_calibration(prediction, calibration),
                    data,
                    list(protocol["reported_horizons"]),
                    station=station,
                ),
            }
        output[f"trailing_{hours}h"] = station_results
    return output


def run_learned(
    name: str,
    seed: int,
    protocol: dict[str, object],
    data,
    device: torch.device,
    directory: Path,
    smoke: bool,
    base_checkpoint: Path | None = None,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    legacy_snapshot = preserve_legacy_partial_run(directory)
    if legacy_snapshot is not None:
        log(
            f"RESTART {name}/seed_{seed} from epoch 1 | legacy partial run preserved at "
            f"{legacy_snapshot} | exact resume was impossible without AdamW/RNG state"
        )
    model = make_model(protocol, data, name, seed, device)
    base_checkpoint_sha256 = None
    if base_checkpoint is not None:
        base_checkpoint_sha256 = initialize_impute_adapter(
            model, base_checkpoint, protocol, seed, device
        )
    train = OutageWindows(data, protocol, "train", 64 if smoke else None)
    validation = OutageWindows(data, protocol, "validation", 512 if smoke else None)
    train_loader = make_loader(train, protocol, True, seed)
    validation_loader = make_loader(validation, protocol, False, seed)
    training = copy.deepcopy(protocol["training"])
    if smoke:
        training.update(epochs=1, patience=1)
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise ValueError(f"No trainable parameters for {name}/seed_{seed}")
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    thresholds = torch.from_numpy(scaled_thresholds(data)).to(device)
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    resume = load_training_checkpoint(
        directory / "last.pt",
        model,
        optimizer,
        train_loader,
        protocol,
        training,
        name,
        seed,
        device,
    )
    if resume is None:
        best, stale, history = float("inf"), 0, []
        best_epoch, start_epoch = 0, 1
        prior_seconds, prior_peak = 0.0, 0
        log(f"FIT {name}/seed_{seed} | train={len(train)} | validation={len(validation)}")
        if base_checkpoint is not None:
            initial_started = time.perf_counter()
            prediction = predict_learned(model, validation_loader, "clean", protocol, device)
            validation_metrics = outage_metrics(
                prediction, data, list(protocol["reported_horizons"])
            )["summary"]
            score = validation_metrics["selection_score"]
            if score is None:
                if not smoke:
                    raise ValueError("Full validation selection requires defined Q95 AP")
                score = float(validation_metrics["scaled_mae"]) + 2 * float(
                    validation_metrics["q95_brier"]
                )
            best = float(score)
            row = {
                "epoch": 0,
                "train_loss": None,
                "selection_score": best,
                "best": True,
                "seconds": time.perf_counter() - initial_started,
                "adapter_initialization": "frozen_impute_then_local_tcn",
                **validation_metrics,
            }
            history.append(row)
            save_checkpoint(directory / "best.pt", model, protocol, name, seed, 0, best)
            current_peak = (
                int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
            )
            save_training_checkpoint(
                directory / "last.pt",
                model,
                optimizer,
                train_loader,
                protocol,
                training,
                name,
                seed,
                0,
                best,
                0,
                0,
                history,
                time.perf_counter() - started,
                current_peak if device.type == "cuda" else None,
                device,
            )
            write_json(directory / "history.json", history)
            log(
                f"{name}/seed_{seed} epoch=0 initialized from frozen imputation base | "
                f"val_MASE={validation_metrics['mase']:.5f} "
                f"Q95_Brier={validation_metrics['q95_brier']:.5f}"
            )
    else:
        best = float(resume["best_selection_score"])
        stale = int(resume["stale_epochs"])
        history = list(resume["history"])
        best_epoch = int(resume["best_epoch"])
        start_epoch = int(resume["epoch"]) + 1
        prior_seconds = float(resume.get("elapsed_training_seconds", 0.0))
        prior_peak = int(resume.get("peak_gpu_memory_bytes") or 0)
        write_json(directory / "history.json", history)
        log(
            f"RESUME {name}/seed_{seed} at epoch={start_epoch} | "
            f"completed_epochs={start_epoch - 1} | best_epoch={best_epoch} | "
            f"stale={stale}/{training['patience']}"
        )
    for epoch in range(start_epoch, int(training["epochs"]) + 1):
        model.train()
        generator = torch.Generator(device=device).manual_seed(seed + epoch * 100003)
        loss_sum, target_count = 0.0, 0
        epoch_started = time.perf_counter()
        for step, batch in enumerate(train_loader, 1):
            batch = causal_outage_inputs(
                on_device(batch, device),
                "train",
                generator,
                fill_limit=int(protocol["fill_limit"]),
                training_probabilities=training["corruption_probabilities"],
            )
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device):
                output = model(batch)
            loss = matched_gaussian_loss(
                output,
                batch["target"],
                batch["target_mask"],
                thresholds,
                training["loss"],
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss for {name}/seed_{seed}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(training["gradient_clip"]), error_if_nonfinite=True
            )
            optimizer.step()
            observed_targets = int(batch["target_mask"].sum())
            loss_sum += float(loss.detach()) * observed_targets
            target_count += observed_targets
            if step % 100 == 0:
                log(
                    f"{name}/seed_{seed} epoch={epoch} batch={step}/{len(train_loader)} "
                    f"loss={loss_sum / max(target_count, 1):.5f}"
                )
        prediction = predict_learned(model, validation_loader, "clean", protocol, device)
        validation_metrics = outage_metrics(prediction, data, list(protocol["reported_horizons"]))[
            "summary"
        ]
        score = validation_metrics["selection_score"]
        if score is None:
            if not smoke:
                raise ValueError("Full validation selection requires defined Q95 AP")
            score = float(validation_metrics["scaled_mae"]) + 2 * float(
                validation_metrics["q95_brier"]
            )
        improved = float(score) < best
        if improved:
            best, stale, best_epoch = float(score), 0, epoch
            save_checkpoint(directory / "best.pt", model, protocol, name, seed, epoch, float(score))
        else:
            stale += 1
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / max(target_count, 1),
            "selection_score": float(score),
            "best": improved,
            "seconds": time.perf_counter() - epoch_started,
            **validation_metrics,
        }
        history.append(row)
        current_peak = (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        )
        save_training_checkpoint(
            directory / "last.pt",
            model,
            optimizer,
            train_loader,
            protocol,
            training,
            name,
            seed,
            epoch,
            best,
            best_epoch,
            stale,
            history,
            prior_seconds + time.perf_counter() - started,
            max(prior_peak, current_peak) if device.type == "cuda" else None,
            device,
        )
        write_json(directory / "history.json", history)
        log(
            f"{name}/seed_{seed} epoch={epoch}/{training['epochs']} "
            f"val_MASE={validation_metrics['mase']:.5f} "
            f"Q95_Brier={validation_metrics['q95_brier']:.5f} "
            f"best_epoch={best_epoch} seconds={row['seconds']:.1f}"
        )
        if stale >= int(training["patience"]):
            break

    checkpoint = torch.load(directory / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    conditions = evaluation_conditions(len(data.stations), smoke)
    metrics, clean_prediction, calibration, trailing_bundle = evaluate(
        lambda condition: predict_learned(model, validation_loader, condition, protocol, device),
        data,
        protocol,
        conditions,
    )
    metrics["natural_comissingness"] = evaluate_natural_comissingness(
        lambda windows: predict_learned(
            model,
            make_loader(windows, protocol, False, seed),
            "clean",
            protocol,
            device,
        ),
        data,
        protocol,
        calibration,
    )
    write_json(directory / "validation_metrics.json", metrics)
    np.savez_compressed(directory / "clean_validation_predictions.npz", **clean_prediction)
    np.savez_compressed(directory / "calibration.npz", factors=calibration)
    np.savez_compressed(directory / "trailing_validation_predictions.npz", **trailing_bundle)
    report_paths = [
        directory / "best.pt",
        directory / "last.pt",
        directory / "history.json",
        directory / "validation_metrics.json",
        directory / "clean_validation_predictions.npz",
        directory / "calibration.npz",
        directory / "trailing_validation_predictions.npz",
    ]
    write_json(
        directory / "complete.json",
        {
            "state": "complete",
            "model": name,
            "seed": seed,
            "best_epoch": best_epoch,
            "best_selection_score": best,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "trainable_parameters": sum(
                parameter.numel() for parameter in model.parameters() if parameter.requires_grad
            ),
            "base_checkpoint_sha256": base_checkpoint_sha256,
            "train_and_validation_seconds": prior_seconds + time.perf_counter() - started,
            "peak_gpu_memory_bytes": (
                max(prior_peak, int(torch.cuda.max_memory_allocated(device)))
                if device.type == "cuda"
                else None
            ),
            "development_only": True,
            "artifact_hashes": {path.name: sha256_file(path) for path in report_paths},
        },
    )


def run_deterministic(
    name: str,
    protocol: dict[str, object],
    data,
    device: torch.device,
    directory: Path,
    smoke: bool,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    train = OutageWindows(data, protocol, "train", 512 if smoke else None)
    validation = OutageWindows(data, protocol, "validation", 512 if smoke else None)
    train_loader = make_loader(train, protocol, False, 0)
    validation_loader = make_loader(validation, protocol, False, 0)
    sigma = fit_deterministic_sigma(name, train_loader, protocol, data, device)
    conditions = evaluation_conditions(len(data.stations), smoke)
    metrics, clean_prediction, calibration, trailing_bundle = evaluate(
        lambda condition: predict_deterministic(
            name, sigma, validation_loader, condition, protocol, device
        ),
        data,
        protocol,
        conditions,
    )
    metrics["natural_comissingness"] = evaluate_natural_comissingness(
        lambda windows: predict_deterministic(
            name,
            sigma,
            make_loader(windows, protocol, False, 0),
            "clean",
            protocol,
            device,
        ),
        data,
        protocol,
        calibration,
    )
    np.savez_compressed(directory / "training_residual_sigma.npz", sigma=sigma.cpu().numpy())
    np.savez_compressed(directory / "clean_validation_predictions.npz", **clean_prediction)
    np.savez_compressed(directory / "calibration.npz", factors=calibration)
    np.savez_compressed(directory / "trailing_validation_predictions.npz", **trailing_bundle)
    write_json(directory / "validation_metrics.json", metrics)
    report_paths = [
        directory / "training_residual_sigma.npz",
        directory / "clean_validation_predictions.npz",
        directory / "calibration.npz",
        directory / "trailing_validation_predictions.npz",
        directory / "validation_metrics.json",
    ]
    write_json(
        directory / "complete.json",
        {
            "state": "complete",
            "model": name,
            "parameters": 0,
            "development_only": True,
            "artifact_hashes": {path.name: sha256_file(path) for path in report_paths},
        },
    )


def verify_complete(directory: Path) -> None:
    marker = read_json(directory / "complete.json")
    if marker.get("state") != "complete":
        raise ValueError(f"Incomplete marker: {directory}")
    for name, expected in marker["artifact_hashes"].items():
        if sha256_file(directory / name) != expected:
            raise ValueError(f"Completed artifact changed: {directory / name}")


def signature(protocol: dict[str, object], prepared_path: Path) -> dict[str, object]:
    sources = [
        PROTOCOL_PATH,
        ROOT / "journal_protocol/development_preprocessing_protocol.json",
        ROOT / "src/aqriskformer/epa_aqs.py",
        ROOT / "src/aqriskformer/outage_data.py",
        ROOT / "src/aqriskformer/outage_evaluation.py",
        ROOT / "src/aqriskformer/outage_models.py",
        ROOT / "src/aqriskformer/models/baselines.py",
        ROOT / "src/aqriskformer/models/graph_baselines.py",
        Path(__file__),
    ]
    result = {
        "protocol": protocol,
        "prepared_sha256": sha256_file(prepared_path),
        "source_hashes": {str(path.relative_to(ROOT)): sha256_file(path) for path in sources},
    }
    result["digest"] = hashlib.sha256(
        json.dumps(result, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return result


def scientifically_compatible_runner_upgrade(
    previous: dict[str, object], current: dict[str, object]
) -> bool:
    """Allow a runner-only reliability patch without mixing scientific revisions."""
    runner_source = Path(__file__).resolve().relative_to(ROOT).as_posix()

    def scientific_part(value: dict[str, object]) -> dict[str, object]:
        source_hashes = {
            Path(str(name)).as_posix(): digest
            for name, digest in dict(value.get("source_hashes", {})).items()
        }
        source_hashes.pop(runner_source, None)
        return {
            "protocol": value.get("protocol"),
            "prepared_sha256": value.get("prepared_sha256"),
            "source_hashes": source_hashes,
        }

    return scientific_part(previous) == scientific_part(current)


def record_runner_upgrade(
    output: Path,
    previous: dict[str, object],
    current: dict[str, object],
) -> None:
    """Record, without rewriting the launch manifest, why continuation is allowed."""
    runner_source = Path(__file__).resolve().relative_to(ROOT).as_posix()

    def runner_hash(value: dict[str, object]) -> object:
        hashes = {
            Path(str(name)).as_posix(): digest
            for name, digest in dict(value.get("source_hashes", {})).items()
        }
        return hashes.get(runner_source)

    path = output / "runner_upgrade_history.json"
    records = read_json(path) if path.exists() else []
    active_digest = str(current["digest"])
    if any(record.get("active_signature_digest") == active_digest for record in records):
        return
    records.append(
        {
            "recorded_utc": utc_now(),
            "reason": "atomic exact-resume reliability patch; scientific protocol unchanged",
            "launch_signature_digest": previous.get("digest"),
            "active_signature_digest": active_digest,
            "previous_runner_sha256": runner_hash(previous),
            "active_runner_sha256": runner_hash(current),
            "scientific_sources_unchanged": True,
            "holdout_accessed": False,
        }
    )
    write_json(path, records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("plan", "smoke", "run"), default="plan")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--learned-models", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--without-deterministic", action="store_true")
    parser.add_argument("--adapter-base-runs", type=Path)
    args = parser.parse_args()
    protocol = read_json(PROTOCOL_PATH)
    if args.stage == "plan":
        print(json.dumps(protocol, indent=2))
        return
    prepared_path = ROOT / protocol["prepared_development"]
    data = load_outage_data(prepared_path, development_only=True)
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.cuda.set_device(device)
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("The locked BF16 protocol requires BF16-capable CUDA hardware")
    training = protocol["training"]
    torch.set_num_threads(int(training["cpu_threads"]))
    torch.set_num_interop_threads(1)
    output = (
        args.output
        or (DEFAULT_OUTPUT / "smoke" if args.stage == "smoke" else DEFAULT_OUTPUT / "full")
    ).resolve()
    output.mkdir(parents=True, exist_ok=True)
    effective = copy.deepcopy(protocol)
    custom_selection = bool(
        args.learned_models
        or args.seeds
        or args.without_deterministic
        or args.adapter_base_runs
    )
    if custom_selection and args.output is None:
        parser.error("Custom model/seed selection requires an explicit --output directory")
    learned_models = list(args.learned_models or effective["learned_models"])
    unknown_models = sorted(set(learned_models) - set(LEARNED_MODELS))
    if unknown_models:
        parser.error(f"Unknown learned models: {', '.join(unknown_models)}")
    deterministic_models = (
        [] if args.without_deterministic else list(effective["deterministic_models"])
    )
    seeds = list(args.seeds) if args.seeds else (
        [42] if args.stage == "smoke" else list(effective["seeds"])
    )
    if len(seeds) != len(set(seeds)):
        parser.error("Seeds must be unique")
    base_checkpoints: dict[int, Path] = {}
    if args.adapter_base_runs is not None:
        adapter_models = {"impute_spatial_adapter", "adaptive_graph_impute_tcn"}
        if len(learned_models) != 1 or learned_models[0] not in adapter_models:
            parser.error(
                "--adapter-base-runs requires exactly one supported adapter model"
            )
        base_root = args.adapter_base_runs.resolve()
        base_hashes = {}
        for seed in seeds:
            base_directory = base_root / "impute_then_local_tcn" / f"seed_{seed}"
            verify_complete(base_directory)
            checkpoint_path = base_directory / "best.pt"
            base_checkpoints[seed] = checkpoint_path
            base_hashes[str(seed)] = sha256_file(checkpoint_path)
        try:
            recorded_base_root = str(base_root.relative_to(ROOT))
        except ValueError:
            recorded_base_root = str(base_root)
        effective["adapter_initialization"] = {
            "base_model": "impute_then_local_tcn",
            "base_runs": recorded_base_root,
            "base_checkpoint_sha256_by_seed": base_hashes,
            "frozen_modules": ["encoder", "local_head"],
            "trainable_module": learned_models[0],
        }
    elif {"impute_spatial_adapter", "adaptive_graph_impute_tcn"}.intersection(
        learned_models
    ):
        parser.error("adapter models require --adapter-base-runs")
    effective["learned_models"] = learned_models
    effective["deterministic_models"] = deterministic_models
    effective["seeds"] = seeds
    expected = len(learned_models) * len(seeds) + len(deterministic_models)
    status_path = output / "status.json"
    with exclusive_lock(output / "runner.lock"):
        current_signature = signature(effective, prepared_path)
        manifest_path = output / "launch_manifest.json"
        if manifest_path.exists():
            previous = read_json(manifest_path)
            if previous["signature"] != current_signature or previous["stage"] != args.stage:
                if previous["stage"] != args.stage or not scientifically_compatible_runner_upgrade(
                    previous["signature"], current_signature
                ):
                    raise ValueError("Code/protocol/data changed; use a new output directory")
                record_runner_upgrade(output, previous["signature"], current_signature)
                log("CONTINUE after audited runner-only reliability upgrade")
        else:
            write_json(
                manifest_path,
                {
                    "stage": args.stage,
                    "created_utc": utc_now(),
                    "signature": current_signature,
                    "environment": environment_snapshot(),
                    "holdout_accessed": False,
                },
            )
        previous_status = read_json(status_path) if status_path.exists() else {}
        status = {
            "state": "running",
            "stage": args.stage,
            "expected_fits": expected,
            "completed_fits": 0,
            "device": str(device),
            "started_utc": previous_status.get("started_utc", utc_now()),
            "resumed_utc": utc_now(),
            "holdout_accessed": False,
        }
        write_json(status_path, status)
        try:
            for name in deterministic_models:
                directory = output / "runs" / name
                if (directory / "complete.json").exists():
                    verify_complete(directory)
                    log(f"SKIP completed {name}")
                else:
                    log(f"RUN deterministic {name}")
                    status.update(current=name, current_state="running", updated_utc=utc_now())
                    write_json(status_path, status)
                    run_deterministic(
                        name, effective, data, device, directory, args.stage == "smoke"
                    )
                status["completed_fits"] += 1
                status.update(current=name, current_state="complete", updated_utc=utc_now())
                write_json(status_path, status)
            for seed_index, seed in enumerate(seeds):
                ordered = learned_models[seed_index:] + learned_models[:seed_index]
                for name in ordered:
                    directory = output / "runs" / name / f"seed_{seed}"
                    if (directory / "complete.json").exists():
                        verify_complete(directory)
                        log(f"SKIP completed {name}/seed_{seed}")
                    else:
                        status.update(
                            current=f"{name}/seed_{seed}",
                            current_state="running",
                            updated_utc=utc_now(),
                        )
                        write_json(status_path, status)
                        run_learned(
                            name,
                            seed,
                            effective,
                            data,
                            device,
                            directory,
                            args.stage == "smoke",
                            base_checkpoints.get(seed),
                        )
                    status["completed_fits"] += 1
                    status.update(
                        current=f"{name}/seed_{seed}",
                        current_state="complete",
                        updated_utc=utc_now(),
                    )
                    write_json(status_path, status)
            status.update(state="complete", completed_utc=utc_now())
            write_json(status_path, status)
            log(f"DEVELOPMENT PROTOCOL COMPLETE | fits={expected} | output={output}")
        except BaseException as error:
            status.update(
                state="failed",
                error=repr(error),
                traceback=traceback.format_exc(),
                updated_utc=utc_now(),
            )
            write_json(status_path, status)
            raise


if __name__ == "__main__":
    main()
