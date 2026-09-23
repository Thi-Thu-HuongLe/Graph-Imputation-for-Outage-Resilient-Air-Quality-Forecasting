"""Leakage-safe helpers for journal-development statistical analysis."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.special import ndtr

from .outage_data import OutageData
from .outage_evaluation import scaled_thresholds


def load_trailing_prediction(path: str | Path, duration_hours: int) -> dict[str, np.ndarray]:
    """Load one duration from the compact trailing-outage prediction bundle."""
    if duration_hours not in (6, 24):
        raise ValueError("Only the prespecified 6 h and 24 h outages are supported")
    prefix = f"h{duration_hours}"
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "origins",
            "horizons",
            f"{prefix}_mu",
            f"{prefix}_sigma",
            f"{prefix}_target",
            f"{prefix}_mask",
        }
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"Prediction archive is missing arrays: {sorted(missing)}")
        prediction = {
            "origins": archive["origins"].copy(),
            "horizons": archive["horizons"].copy(),
            "mu": archive[f"{prefix}_mu"].copy(),
            "sigma": archive[f"{prefix}_sigma"].copy(),
            "target": archive[f"{prefix}_target"].copy(),
            "mask": archive[f"{prefix}_mask"].copy(),
        }
    shape = prediction["mu"].shape
    if any(prediction[key].shape != shape for key in ("sigma", "target", "mask")):
        raise ValueError("Prediction arrays have inconsistent shapes")
    if shape[0] != prediction["origins"].size or shape[1] != prediction["horizons"].size:
        raise ValueError("Origin or horizon metadata does not match prediction arrays")
    if not np.all(np.diff(prediction["origins"]) > 0):
        raise ValueError("Forecast origins must be strictly increasing")
    return prediction


def apply_pollutant_scale(
    prediction: dict[str, np.ndarray], factors: np.ndarray
) -> dict[str, np.ndarray]:
    """Return a prediction copy with per-pollutant Gaussian scale factors."""
    factors = np.asarray(factors, dtype=np.float32)
    pollutants = prediction["sigma"].shape[-1]
    if factors.shape != (pollutants,) or not np.isfinite(factors).all() or not (factors > 0).all():
        raise ValueError("Invalid pollutant calibration factors")
    return {
        **prediction,
        "sigma": np.asarray(prediction["sigma"], dtype=np.float32)
        * factors[None, None, None, :],
    }


def verify_paired_predictions(
    prediction_a: dict[str, np.ndarray], prediction_b: dict[str, np.ndarray]
) -> None:
    """Reject comparisons that are not paired on metadata, targets, and masks."""
    for key in ("origins", "horizons", "target", "mask"):
        if not np.array_equal(prediction_a[key], prediction_b[key]):
            raise ValueError(f"Paired predictions differ in '{key}'")


def origin_macro_losses(
    prediction: dict[str, np.ndarray], data: OutageData
) -> dict[str, np.ndarray]:
    """Compute cell-paired losses, then macro-average valid cells at each origin.

    MASE scaling is station/pollutant specific. Q95 Brier uses the locked training-only
    Q95 threshold. The result retains one value per forecast origin so temporal
    resampling never treats horizon/station/pollutant cells as independent samples.
    """
    mu = np.asarray(prediction["mu"], dtype=float)
    sigma = np.asarray(prediction["sigma"], dtype=float)
    target = np.asarray(prediction["target"], dtype=float)
    mask = np.asarray(prediction["mask"], dtype=bool)
    if not (np.isfinite(mu).all() and np.isfinite(sigma).all() and (sigma > 0).all()):
        raise ValueError("Nonfinite predictions or nonpositive Gaussian scales")
    pollutants = data.n_pollutants
    if mu.shape[-2:] != (len(data.stations), pollutants):
        raise ValueError("Prediction station/pollutant dimensions do not match data")

    mase_multiplier = (
        data.scale[:, :pollutants] / data.mase
    )[None, None, :, :]
    mase = np.abs(target - mu) * mase_multiplier
    q95 = scaled_thresholds(data)[..., 2][None, None, :, :]
    probability = ndtr((mu - q95) / sigma)
    brier = (probability - (target > q95)) ** 2

    def masked_macro(values: np.ndarray) -> np.ndarray:
        axes = tuple(range(1, values.ndim))
        count = mask.sum(axis=axes)
        total = np.where(mask, values, 0.0).sum(axis=axes)
        return np.divide(
            total,
            count,
            out=np.full(count.shape, np.nan, dtype=float),
            where=count > 0,
        )

    return {"mase": masked_macro(mase), "q95_brier": masked_macro(brier)}


def hourly_loss_grid(origins: np.ndarray, losses: np.ndarray) -> np.ndarray:
    """Place origin losses on their regular hourly grid without closing gaps."""
    origins = np.asarray(origins, dtype=np.int64)
    losses = np.asarray(losses, dtype=float)
    if origins.ndim != 1 or losses.shape != origins.shape or origins.size == 0:
        raise ValueError("Origins and losses must be nonempty aligned vectors")
    if not np.all(np.diff(origins) > 0):
        raise ValueError("Forecast origins must be strictly increasing")
    grid = np.full(int(origins[-1] - origins[0] + 1), np.nan, dtype=float)
    grid[origins - origins[0]] = losses
    return grid


def finite_column_mean(values: list[np.ndarray]) -> np.ndarray:
    """Average aligned seed vectors while retaining all-missing hourly slots."""
    stacked = np.stack(values)
    valid = np.isfinite(stacked)
    count = valid.sum(axis=0)
    total = np.where(valid, stacked, 0.0).sum(axis=0)
    return np.divide(
        total,
        count,
        out=np.full(count.shape, np.nan, dtype=float),
        where=count > 0,
    )
