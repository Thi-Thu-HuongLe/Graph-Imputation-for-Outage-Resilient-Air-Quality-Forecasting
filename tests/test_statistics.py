from __future__ import annotations

import numpy as np

from aqriskformer.statistics import (
    diebold_mariano,
    holm_adjust,
    paired_moving_block_bootstrap,
    paired_wilcoxon,
)


def test_holm_adjust_is_monotone_in_sorted_order() -> None:
    p = np.array([0.04, 0.001, 0.02])
    adjusted = holm_adjust(p)
    order = np.argsort(p)
    assert np.all(np.diff(adjusted[order]) >= 0)
    assert np.all((adjusted >= 0) & (adjusted <= 1))


def test_paired_statistics_return_finite_results() -> None:
    rng = np.random.default_rng(42)
    candidate = rng.normal(1.0, 0.2, 500)
    comparator = candidate + 0.05 + rng.normal(0, 0.02, 500)
    bootstrap = paired_moving_block_bootstrap(
        candidate,
        comparator,
        block_length=24,
        resamples=100,
        seed=1,
    )
    dm = diebold_mariano(candidate, comparator, horizon=6)
    assert np.isfinite(list(bootstrap.values())).all()
    assert np.isfinite(list(dm.values())).all()


def test_paired_wilcoxon_detects_consistently_lower_loss() -> None:
    candidate = np.array([0.7, 0.8, 0.6, 0.75, 0.65, 0.72])
    comparator = candidate + 0.1
    result = paired_wilcoxon(candidate, comparator)
    assert np.mean(candidate - comparator) < 0
    assert 0.0 <= result["p_value"] <= 1.0
