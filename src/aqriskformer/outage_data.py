"""Leakage-safe windows and input outages for the journal comparison protocol."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .data import PreparedAirQuality


@dataclass
class OutageData:
    raw: np.ndarray
    observed: np.ndarray
    calendar: np.ndarray
    center: np.ndarray
    scale: np.ndarray
    thresholds: np.ndarray
    mase: np.ndarray
    timestamps: np.ndarray
    stations: list[str]
    pollutants: list[str]
    graph: np.ndarray
    split_bounds: dict[str, tuple[int, int]]

    @property
    def n_features(self) -> int:
        return self.raw.shape[-1]

    @property
    def n_pollutants(self) -> int:
        return len(self.pollutants)


def load_outage_data(path: str | Path, *, development_only: bool = True) -> OutageData:
    prepared = PreparedAirQuality.load(path)
    if development_only and "test" in prepared.split_bounds:
        raise ValueError("Development loader refuses an artifact containing a test split")
    if not {"train", "validation"}.issubset(prepared.split_bounds):
        raise ValueError("Prepared data must define train and validation splits")
    raw = np.where(prepared.observed_mask, prepared.values, 0.0).astype(np.float32)
    result = OutageData(
        raw=raw,
        observed=prepared.observed_mask,
        calendar=prepared.calendar,
        center=prepared.center,
        scale=prepared.scale,
        thresholds=prepared.risk_thresholds,
        mase=prepared.mase_scale24,
        timestamps=prepared.timestamps_ns,
        stations=prepared.stations,
        pollutants=prepared.pollutants,
        graph=prepared.train_correlation_graph,
        split_bounds=prepared.split_bounds,
    )
    validate_outage_data(result)
    return result


def validate_outage_data(data: OutageData) -> None:
    if data.raw.shape != data.observed.shape:
        raise ValueError("Prepared values and masks have different shapes")
    if data.raw.shape[:2] != (len(data.timestamps), len(data.stations)):
        raise ValueError("Prepared time/station dimensions are inconsistent")
    if data.n_pollutants < 1 or data.raw.shape[-1] < data.n_pollutants:
        raise ValueError("Pollutant metadata is inconsistent with the feature tensor")
    if data.center.shape != data.scale.shape or data.center.shape != data.raw.shape[1:]:
        raise ValueError("Scaling statistics have invalid dimensions")
    if data.mase.shape != (len(data.stations), data.n_pollutants):
        raise ValueError("MASE scale has invalid dimensions")
    if data.thresholds.shape[0] != data.n_pollutants:
        raise ValueError("Thresholds have invalid pollutant dimension")
    if data.graph.shape != (len(data.stations), len(data.stations)):
        raise ValueError("Station graph has invalid dimensions")
    train_start, train_end = data.split_bounds["train"]
    validation_start, validation_end = data.split_bounds["validation"]
    if not (
        0 <= train_start <= train_end < validation_start <= validation_end < len(data.timestamps)
    ):
        raise ValueError("Train and validation bounds must be chronological and disjoint")
    for array in (data.raw, data.calendar, data.center, data.scale, data.thresholds, data.mase):
        if not np.isfinite(array).all():
            raise ValueError("Prepared development tensor contains nonfinite values")
    if not (data.scale > 0).all() or not (data.mase > 0).all():
        raise ValueError("Scale and MASE denominators must be positive")


class OutageWindows(Dataset):
    def __init__(
        self,
        data: OutageData,
        protocol: dict[str, object],
        split: str,
        max_windows: int | None = None,
    ) -> None:
        if split not in data.split_bounds:
            raise ValueError(f"Unknown or unavailable split: {split}")
        self.data = data
        self.length = int(protocol["lookback"])
        self.horizon = int(protocol["horizon"])
        start, end = data.split_bounds[split]
        first = max(self.length - 1, start - 1)
        last = end - self.horizon
        stride = int(protocol[f"{split}_stride"])
        self.origins = np.arange(first, last + 1, stride)
        if max_windows is not None:
            self.origins = self.origins[:max_windows]
        if not len(self.origins):
            raise ValueError(f"No complete {split} windows")

    def __len__(self) -> int:
        return len(self.origins)

    def __getitem__(self, index: int) -> dict[str, np.ndarray | int]:
        origin = int(self.origins[index])
        past = slice(origin - self.length + 1, origin + 1)
        future = slice(origin + 1, origin + 1 + self.horizon)
        targets = slice(0, self.data.n_pollutants)
        return {
            "raw": self.data.raw[past],
            "mask": self.data.observed[past],
            "calendar": self.data.calendar[past],
            "target": self.data.raw[future, :, targets],
            "target_mask": self.data.observed[future, :, targets],
            "origin": origin,
        }


def _erase_trailing_station(mask: torch.Tensor, station: int, hours: int) -> None:
    if not 0 <= station < mask.shape[2]:
        raise ValueError("Invalid target station")
    if not 0 < hours <= mask.shape[1]:
        raise ValueError("Invalid trailing-outage duration")
    mask[:, -hours:, station, :] = False


def natural_comissingness_origins(
    data: OutageData,
    origins: np.ndarray,
    minimum_hours: int,
    core_pollutants: tuple[str, ...] = ("PM2.5", "NO2", "O3"),
    require_next_hour_recovery: bool = True,
) -> np.ndarray:
    """Identify forecast origins preceded by simultaneous core-pollutant gaps.

    This is an observable co-missingness definition, not a claim that the cause is
    a verified hardware or communications failure.
    """
    if minimum_hours < 1:
        raise ValueError("minimum_hours must be positive")
    if np.any(origins < 0) or np.any(origins >= len(data.timestamps)):
        raise ValueError("Forecast origin lies outside the data tensor")
    indices = [data.pollutants.index(name) for name in core_pollutants]
    core_observed = data.observed[:, :, indices].any(axis=-1)
    missing_all = ~core_observed
    run_length = np.zeros(missing_all.shape, dtype=np.int32)
    for time in range(len(run_length)):
        previous = run_length[time - 1] if time else 0
        run_length[time] = np.where(missing_all[time], previous + 1, 0)
    selected = run_length[origins] >= minimum_hours
    network_available = core_observed[origins].sum(axis=1, keepdims=True)
    other_station_available = network_available - core_observed[origins].astype(np.int32) > 0
    selected &= other_station_available
    if require_next_hour_recovery:
        if np.any(origins + 1 >= len(data.timestamps)):
            raise ValueError("Recovery check requires an hour after every origin")
        selected &= core_observed[origins + 1]
    return selected


def causal_outage_inputs(
    batch: dict[str, torch.Tensor],
    condition: str,
    generator: torch.Generator,
    *,
    fill_limit: int,
    training_probabilities: dict[str, float] | None = None,
) -> dict[str, torch.Tensor]:
    """Erase observations, then recompute causal values and observation ages."""
    mask = batch["mask"].clone().bool()
    batch_size, length, stations, _ = mask.shape
    device = mask.device
    time = torch.arange(length, device=device)[None, :, None, None]
    kinds = torch.zeros(batch_size, device=device, dtype=torch.long)

    if condition == "train":
        if training_probabilities is None:
            raise ValueError("Training corruption probabilities are required")
        required = ("clean", "block_6h", "block_24h", "station_dropout")
        probabilities = torch.tensor(
            [float(training_probabilities[key]) for key in required], device=device
        )
        if not torch.isclose(probabilities.sum(), probabilities.new_tensor(1.0)):
            raise ValueError("Training corruption probabilities must sum to one")
        draw = torch.rand(batch_size, device=device, generator=generator)
        boundaries = probabilities.cumsum(0)
        kinds = torch.bucketize(draw, boundaries[:-1])
    elif condition == "clean":
        pass
    elif condition in ("random_block_6h", "random_block_24h"):
        kinds.fill_(1 if condition.endswith("6h") else 2)
    elif condition.startswith("station_full_"):
        station = int(condition.removeprefix("station_full_"))
        if not 0 <= station < stations:
            raise ValueError("Invalid dropped station")
        mask[:, :, station, :] = False
    elif condition.startswith("station_trailing_"):
        parts = condition.split("_")
        if len(parts) != 4 or not parts[2].endswith("h"):
            raise ValueError(f"Malformed trailing-outage condition: {condition}")
        _erase_trailing_station(mask, int(parts[3]), int(parts[2][:-1]))
    else:
        raise ValueError(f"Unknown condition: {condition}")

    for kind, block in ((1, 6), (2, 24)):
        if block > length:
            raise ValueError("Lookback shorter than corruption block")
        starts = torch.randint(
            length - block + 1,
            (batch_size, 1, stations, 1),
            device=device,
            generator=generator,
        )
        erased = (time >= starts) & (time < starts + block)
        erased &= kinds[:, None, None, None] == kind
        mask &= ~erased

    training_station_dropout = kinds == 3
    if training_station_dropout.any():
        dropped = torch.randint(stations, (batch_size,), device=device, generator=generator)
        station_indices = torch.arange(stations, device=device)[None, None, :, None]
        selected = station_indices == dropped[:, None, None, None]
        mask &= ~(selected & training_station_dropout[:, None, None, None])

    last = torch.where(mask, time, -1).cummax(dim=1).values
    age = time - last
    gathered = batch["raw"].gather(1, last.clamp_min(0).expand_as(batch["raw"]))
    values = torch.where((last >= 0) & (age <= fill_limit), gathered, 0.0)
    return {**batch, "values": values, "mask": mask, "gaps": age.float().log1p()}
