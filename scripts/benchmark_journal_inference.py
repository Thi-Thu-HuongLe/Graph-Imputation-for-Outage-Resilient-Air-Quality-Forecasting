"""Standardized CPU/GPU inference benchmark for the frozen holdout models."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_journal_development import autocast_context, make_model, on_device
from run_journal_holdout import runtime_protocol

from aqriskformer.epa_aqs_holdout import (
    load_holdout_outage_data,
    verify_execution_freeze,
)
from aqriskformer.outage_data import OutageWindows, causal_outage_inputs
from aqriskformer.outage_models import deterministic_forecast
from aqriskformer.utils import (
    environment_snapshot,
    read_json,
    sha256_file,
    utc_now,
    write_json,
)

PROTOCOL_PATH = ROOT / "journal_protocol/final_holdout_protocol.json"
FREEZE_PATH = ROOT / "journal_protocol/final_holdout_execution_freeze.json"


def log(message: str) -> None:
    print(f"[{utc_now()}] {message}", flush=True)


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def prepared_batch(
    windows: OutageWindows,
    protocol: dict[str, object],
    batch_size: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    subset = Subset(windows, range(batch_size))
    batch = next(iter(DataLoader(subset, batch_size=batch_size, shuffle=False)))
    generator = torch.Generator(device=device).manual_seed(int(protocol["corruption_seed"]))
    return causal_outage_inputs(
        on_device(batch, device),
        "station_trailing_24h_0",
        generator,
        fill_limit=int(protocol["fill_limit"]),
    )


def measure(
    function,
    device: torch.device,
    warmup: int,
    repetitions: int,
) -> tuple[np.ndarray, int | None]:
    with torch.inference_mode():
        for _ in range(warmup):
            function()
        synchronize(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        durations = np.empty(repetitions, dtype=float)
        for index in range(repetitions):
            synchronize(device)
            started = time.perf_counter_ns()
            function()
            synchronize(device)
            durations[index] = (time.perf_counter_ns() - started) / 1e6
        peak = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    return durations, peak


def summarize(
    model: str,
    device: torch.device,
    parameters: int,
    batch_size: int,
    durations: np.ndarray,
    peak: int | None,
    warmup: int,
) -> dict[str, object]:
    mean_ms = float(durations.mean())
    return {
        "model": model,
        "device": str(device),
        "batch_size": batch_size,
        "condition": "station_trailing_24h_station_0",
        "precision": "bf16_autocast" if device.type == "cuda" else "float32",
        "cpu_threads": torch.get_num_threads() if device.type == "cpu" else None,
        "warmup_repetitions": warmup,
        "timed_repetitions": len(durations),
        "parameters": parameters,
        "latency_mean_ms_per_batch": mean_ms,
        "latency_median_ms_per_batch": float(np.median(durations)),
        "latency_p95_ms_per_batch": float(np.quantile(durations, 0.95)),
        "throughput_windows_per_second": float(1000.0 * batch_size / mean_ms),
        "peak_gpu_memory_mib": None if peak is None else peak / 2**20,
    }


def source_directory(final: dict[str, object], model: str, seed: int) -> Path:
    key = "candidate" if model == final["candidate"] else "comparators"
    base = project_path(final["checkpoint_roots"][key]) / model
    return base if model in final["deterministic_comparators"] else base / f"seed_{seed}"


def benchmark_model(
    final: dict[str, object],
    protocol: dict[str, object],
    data,
    windows: OutageWindows,
    model_name: str,
    seed: int,
    device: torch.device,
    warmup: int,
    repetitions: int,
) -> list[dict[str, object]]:
    deterministic = model_name in final["deterministic_comparators"]
    source = source_directory(final, model_name, seed)
    if deterministic:
        with np.load(source / "training_residual_sigma.npz", allow_pickle=False) as archive:
            sigma = torch.from_numpy(archive["sigma"].copy()).to(device)
        parameters = 0

        def forward(batch):
            return deterministic_forecast(model_name, batch, sigma)

    else:
        checkpoint = torch.load(source / "best.pt", map_location=device, weights_only=False)
        model = make_model(checkpoint["protocol"], data, model_name, seed, device)
        model.load_state_dict(checkpoint["model"], strict=True)
        model.eval()
        parameters = sum(parameter.numel() for parameter in model.parameters())

        def forward(batch):
            with autocast_context(device):
                return model(batch)

    rows = []
    for batch_size in (1, 32):
        batch = prepared_batch(windows, protocol, batch_size, device)
        durations, peak = measure(
            lambda frozen_batch=batch: forward(frozen_batch),
            device,
            warmup,
            repetitions,
        )
        rows.append(
            summarize(
                model_name,
                device,
                parameters,
                batch_size,
                durations,
                peak,
                warmup,
            )
        )
        log(
            f"{model_name} {device} batch={batch_size} "
            f"median={rows[-1]['latency_median_ms_per_batch']:.3f} ms "
            f"throughput={rows[-1]['throughput_windows_per_second']:.2f} windows/s"
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--devices", nargs="+", default=["cpu", "cuda:0"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.warmup < 1 or args.repetitions < 2 or args.cpu_threads < 1:
        raise ValueError("Warmup, repetitions, and CPU threads must be positive")
    verify_execution_freeze(ROOT, FREEZE_PATH)
    final = read_json(PROTOCOL_PATH)
    data = load_holdout_outage_data(project_path(final["data"]["prepared_holdout"]))
    protocol = runtime_protocol(final)
    windows = OutageWindows(data, protocol, "test")
    models = [
        final["candidate"],
        *final["learned_comparators"],
        *final["deterministic_comparators"],
    ]
    prior_threads = torch.get_num_threads()
    torch.set_num_threads(args.cpu_threads)
    rows: list[dict[str, object]] = []
    try:
        for device_name in args.devices:
            device = torch.device(device_name)
            if device.type == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("CUDA benchmark requested but CUDA is unavailable")
            for model in models:
                log(f"benchmarking {model} on {device}")
                rows.extend(
                    benchmark_model(
                        final,
                        protocol,
                        data,
                        windows,
                        model,
                        args.seed,
                        device,
                        args.warmup,
                        args.repetitions,
                    )
                )
    finally:
        torch.set_num_threads(prior_threads)

    output = project_path(final["output"]) / "computational_benchmark"
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    csv_path = output / "inference_benchmark.csv"
    frame.to_csv(csv_path, index=False)
    protocol_record = {
        "created_utc": utc_now(),
        "post_confirmatory_characterization": True,
        "model_or_calibration_changed": False,
        "holdout_targets_used": False,
        "condition": "24 h trailing outage at station index 0",
        "seed": args.seed,
        "batch_sizes": [1, 32],
        "warmup_repetitions": args.warmup,
        "timed_repetitions": args.repetitions,
        "cpu_threads": args.cpu_threads,
        "gpu_precision": "bf16 autocast",
        "cpu_precision": "float32",
        "timing_scope": "model forward pass only; data loading and host-device transfer excluded",
        "environment": environment_snapshot(),
        "source_sha256": sha256_file(Path(__file__).resolve()),
        "results_sha256": sha256_file(csv_path),
    }
    write_json(output / "benchmark_protocol.json", protocol_record)
    log(f"benchmark complete: {csv_path}")


if __name__ == "__main__":
    main()
