from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from aqriskformer.development_analysis import (
    apply_pollutant_scale,
    finite_column_mean,
    hourly_loss_grid,
    origin_macro_losses,
    verify_paired_predictions,
)


def prediction() -> dict[str, np.ndarray]:
    return {
        "origins": np.array([10, 16]),
        "horizons": np.array([1]),
        "mu": np.zeros((2, 1, 1, 1), dtype=np.float32),
        "sigma": np.ones((2, 1, 1, 1), dtype=np.float32),
        "target": np.array([1.0, 2.0], dtype=np.float32).reshape(2, 1, 1, 1),
        "mask": np.ones((2, 1, 1, 1), dtype=bool),
    }


def test_hourly_loss_grid_retains_six_hour_gap() -> None:
    grid = hourly_loss_grid(np.array([10, 16]), np.array([1.0, 2.0]))
    assert grid.size == 7
    assert grid[[0, 6]].tolist() == [1.0, 2.0]
    assert np.isnan(grid[1:6]).all()


def test_paired_predictions_reject_different_targets() -> None:
    first = prediction()
    second = {**prediction(), "target": prediction()["target"] + 1}
    with pytest.raises(ValueError, match="target"):
        verify_paired_predictions(first, second)


def test_origin_macro_losses_applies_station_mase_scale() -> None:
    data = SimpleNamespace(
        n_pollutants=1,
        stations=["A"],
        scale=np.array([[2.0]]),
        mase=np.array([[4.0]]),
        thresholds=np.array([[0.0, 0.5, 1.0]]),
        center=np.array([[0.0]]),
    )
    losses = origin_macro_losses(prediction(), data)
    np.testing.assert_allclose(losses["mase"], [0.5, 1.0])
    assert np.isfinite(losses["q95_brier"]).all()


def test_pollutant_scale_and_finite_seed_mean() -> None:
    calibrated = apply_pollutant_scale(prediction(), np.array([2.0]))
    np.testing.assert_allclose(calibrated["sigma"], 2.0)
    result = finite_column_mean(
        [np.array([1.0, np.nan, 3.0]), np.array([3.0, np.nan, 5.0])]
    )
    np.testing.assert_allclose(result[[0, 2]], [2.0, 4.0])
    assert np.isnan(result[1])
