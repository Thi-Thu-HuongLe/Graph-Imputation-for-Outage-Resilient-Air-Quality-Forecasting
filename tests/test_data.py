from __future__ import annotations

import numpy as np

from aqriskformer.data import _fit_robust_scaler, _time_since_observation


def test_scaler_ignores_validation_and_test_values() -> None:
    values = np.array([1.0, 2.0, 3.0, 10_000.0], dtype=np.float32).reshape(4, 1, 1)
    observed = np.ones_like(values, dtype=bool)
    train_mask = np.array([True, True, True, False])
    center, scale = _fit_robust_scaler(values, observed, train_mask)
    assert center.item() == 2.0
    assert scale.item() == 1.0


def test_time_gap_resets_only_on_observation() -> None:
    observed = np.array([True, False, False, True, False]).reshape(5, 1, 1)
    gaps = _time_since_observation(observed)
    np.testing.assert_array_equal(gaps[:, 0, 0], [0, 1, 2, 0, 1])
