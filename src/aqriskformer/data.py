"""Prepared EPA AQS tensor schema and causal preprocessing utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd

QUANTILES = (0.80, 0.90, 0.95)


@dataclass
class PreparedAirQuality:
    name: str
    timestamps_ns: np.ndarray
    stations: list[str]
    pollutants: list[str]
    feature_names: list[str]
    meteorology: list[str]
    values: np.ndarray
    observed_mask: np.ndarray
    time_gaps: np.ndarray
    calendar: np.ndarray
    station_static: np.ndarray
    native_pollutants: np.ndarray
    target_mask: np.ndarray
    center: np.ndarray
    scale: np.ndarray
    risk_thresholds: np.ndarray
    mase_scale24: np.ndarray
    train_correlation_graph: np.ndarray
    split_bounds: dict[str, tuple[int, int]]

    @cached_property
    def training_event_rates(self) -> np.ndarray:
        """Station/pollutant/threshold climatology from observed TRAIN targets."""
        start, end = self.split_bounds["train"]
        y = self.native_pollutants[start : end + 1]
        observed = self.target_mask[start : end + 1] & np.isfinite(y)
        counts = observed.sum(axis=0)[..., None]
        events = (y[..., None] > self.risk_thresholds[None, None]) & observed[..., None]
        return np.divide(
            events.sum(axis=0),
            counts,
            out=np.full(events.shape[1:], np.nan, dtype=float),
            where=counts > 0,
        )

    @property
    def timestamps(self) -> pd.DatetimeIndex:
        return pd.to_datetime(self.timestamps_ns)

    @property
    def n_pollutants(self) -> int:
        return len(self.pollutants)

    @property
    def n_stations(self) -> int:
        return len(self.stations)

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "name": self.name,
            "stations": self.stations,
            "pollutants": self.pollutants,
            "feature_names": self.feature_names,
            "meteorology": self.meteorology,
            "split_bounds": self.split_bounds,
        }
        np.savez_compressed(
            target,
            timestamps_ns=self.timestamps_ns.astype("datetime64[ns]"),
            values=self.values.astype(np.float32),
            observed_mask=self.observed_mask.astype(np.uint8),
            time_gaps=self.time_gaps.astype(np.float32),
            calendar=self.calendar.astype(np.float32),
            station_static=self.station_static.astype(np.float32),
            native_pollutants=self.native_pollutants.astype(np.float32),
            target_mask=self.target_mask.astype(np.uint8),
            center=self.center.astype(np.float32),
            scale=self.scale.astype(np.float32),
            risk_thresholds=self.risk_thresholds.astype(np.float32),
            mase_scale24=self.mase_scale24.astype(np.float32),
            train_correlation_graph=self.train_correlation_graph.astype(np.float32),
            metadata=np.array(json.dumps(metadata)),
        )

    @classmethod
    def load(cls, path: str | Path) -> PreparedAirQuality:
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"].item()))
            return cls(
                name=metadata["name"],
                timestamps_ns=archive["timestamps_ns"],
                stations=list(metadata["stations"]),
                pollutants=list(metadata["pollutants"]),
                feature_names=list(metadata["feature_names"]),
                meteorology=list(metadata["meteorology"]),
                values=archive["values"],
                observed_mask=archive["observed_mask"].astype(bool),
                time_gaps=archive["time_gaps"],
                calendar=archive["calendar"],
                station_static=archive["station_static"],
                native_pollutants=archive["native_pollutants"],
                target_mask=archive["target_mask"].astype(bool),
                center=archive["center"],
                scale=archive["scale"],
                risk_thresholds=archive["risk_thresholds"],
                mase_scale24=archive["mase_scale24"],
                train_correlation_graph=archive["train_correlation_graph"],
                split_bounds={k: tuple(v) for k, v in metadata["split_bounds"].items()},
            )


def _timestamp_mask(index: pd.DatetimeIndex, bounds: list[str]) -> np.ndarray:
    start, end = pd.Timestamp(bounds[0]), pd.Timestamp(bounds[1])
    return np.asarray((index >= start) & (index <= end))


def _split_index_bounds(index: pd.DatetimeIndex, bounds: list[str]) -> tuple[int, int]:
    mask = _timestamp_mask(index, bounds)
    positions = np.flatnonzero(mask)
    if positions.size == 0:
        raise ValueError(f"Split {bounds} does not overlap dataset")
    return int(positions[0]), int(positions[-1])


def _fit_robust_scaler(
    values: np.ndarray, observed: np.ndarray, train_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    train = np.where(observed[train_mask], values[train_mask], np.nan)
    center = np.nanmedian(train, axis=0)
    q75 = np.nanpercentile(train, 75, axis=0)
    q25 = np.nanpercentile(train, 25, axis=0)
    scale = q75 - q25
    global_center = np.nanmedian(train, axis=(0, 1))
    global_scale = np.nanpercentile(train, 75, axis=(0, 1)) - np.nanpercentile(
        train, 25, axis=(0, 1)
    )
    center = np.where(np.isfinite(center), center, global_center[None, :])
    scale = np.where(np.isfinite(scale) & (scale > 1e-8), scale, global_scale[None, :])
    scale = np.where(np.isfinite(scale) & (scale > 1e-8), scale, 1.0)
    return center.astype(np.float32), scale.astype(np.float32)


def _causal_fill(values: np.ndarray, train_center: np.ndarray, limit: int) -> np.ndarray:
    output = values.copy()
    for station in range(values.shape[1]):
        for feature in range(values.shape[2]):
            series = pd.Series(values[:, station, feature])
            output[:, station, feature] = (
                series.ffill(limit=limit).fillna(float(train_center[station, feature])).to_numpy()
            )
    return output


def _time_since_observation(observed: np.ndarray) -> np.ndarray:
    gaps = np.zeros(observed.shape, dtype=np.float32)
    for station in range(observed.shape[1]):
        for feature in range(observed.shape[2]):
            elapsed = 0.0
            for t in range(observed.shape[0]):
                if observed[t, station, feature]:
                    elapsed = 0.0
                else:
                    elapsed += 1.0
                gaps[t, station, feature] = elapsed
    return gaps


def _calendar_features(index: pd.DatetimeIndex) -> np.ndarray:
    hour = index.hour.to_numpy()
    weekday = index.dayofweek.to_numpy()
    dayofyear = index.dayofyear.to_numpy()
    weekend = (weekday >= 5).astype(np.float32)
    holiday = np.zeros(len(index), dtype=np.float32)
    return np.column_stack(
        [
            np.sin(2 * np.pi * hour / 24),
            np.cos(2 * np.pi * hour / 24),
            np.sin(2 * np.pi * weekday / 7),
            np.cos(2 * np.pi * weekday / 7),
            np.sin(2 * np.pi * dayofyear / 365.25),
            np.cos(2 * np.pi * dayofyear / 365.25),
            weekend,
            holiday,
        ]
    ).astype(np.float32)


def _risk_thresholds(
    values: np.ndarray, observed: np.ndarray, train_mask: np.ndarray
) -> np.ndarray:
    output = np.empty((values.shape[2], len(QUANTILES)), dtype=np.float32)
    for pollutant in range(values.shape[2]):
        valid = values[train_mask, :, pollutant][observed[train_mask, :, pollutant]]
        output[pollutant] = np.quantile(valid, QUANTILES)
    return output


def _mase_scale(
    values: np.ndarray, observed: np.ndarray, train_mask: np.ndarray, season: int
) -> np.ndarray:
    train_indices = np.flatnonzero(train_mask)
    start, end = train_indices[0], train_indices[-1] + 1
    output = np.ones((values.shape[1], values.shape[2]), dtype=np.float32)
    for station in range(values.shape[1]):
        for pollutant in range(values.shape[2]):
            current = values[start + season : end, station, pollutant]
            lagged = values[start : end - season, station, pollutant]
            valid = (
                observed[start + season : end, station, pollutant]
                & observed[start : end - season, station, pollutant]
            )
            denominator = np.mean(np.abs(current[valid] - lagged[valid])) if valid.any() else np.nan
            output[station, pollutant] = (
                denominator if np.isfinite(denominator) and denominator > 1e-8 else 1.0
            )
    return output


def _normalize_static(static: np.ndarray) -> np.ndarray:
    if static.size == 0 or np.allclose(static, 0):
        return np.zeros_like(static, dtype=np.float32)
    center = np.nanmean(static, axis=0, keepdims=True)
    scale = np.nanstd(static, axis=0, keepdims=True)
    scale = np.where(scale > 1e-8, scale, 1.0)
    return np.nan_to_num((static - center) / scale).astype(np.float32)

