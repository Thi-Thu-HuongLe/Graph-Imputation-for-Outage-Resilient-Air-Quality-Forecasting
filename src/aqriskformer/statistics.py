from __future__ import annotations

import numpy as np
from scipy import stats


def paired_moving_block_bootstrap(
    loss_a: np.ndarray,
    loss_b: np.ndarray,
    *,
    block_length: int = 168,
    resamples: int = 10_000,
    seed: int = 42,
) -> dict[str, float]:
    difference = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    valid = np.isfinite(difference)
    if not valid.any():
        raise ValueError("No finite paired losses")
    if resamples < 1:
        raise ValueError("resamples must be positive")
    rng = np.random.default_rng(seed)
    block_length = min(block_length, difference.size)
    if block_length < 1:
        raise ValueError("block_length must be positive")
    n_starts = difference.size - block_length + 1
    blocks_needed = int(np.ceil(difference.size / block_length))
    estimates = np.empty(resamples)
    # Array positions are hourly slots. Never close gaps before resampling.
    prefix = np.concatenate(([0.0], np.cumsum(np.where(valid, difference, 0.0))))
    count_prefix = np.concatenate(([0], np.cumsum(valid)))
    full_block_sums = prefix[block_length:] - prefix[:-block_length]
    full_block_counts = count_prefix[block_length:] - count_prefix[:-block_length]
    final_block_length = difference.size - (blocks_needed - 1) * block_length
    partial_block_sums = (
        prefix[final_block_length : final_block_length + n_starts] - prefix[:n_starts]
    )
    partial_block_counts = (
        count_prefix[final_block_length : final_block_length + n_starts] - count_prefix[:n_starts]
    )

    # Work with block sums instead of materializing every resampled time series.
    # Chunking keeps memory bounded for the 10,000-resample paper protocol.
    chunk_size = 4096
    for start_index in range(0, resamples, chunk_size):
        stop_index = min(resamples, start_index + chunk_size)
        starts = rng.integers(
            0,
            n_starts,
            size=(stop_index - start_index, blocks_needed),
        )
        totals = partial_block_sums[starts[:, -1]].copy()
        counts = partial_block_counts[starts[:, -1]].copy()
        if blocks_needed > 1:
            totals += full_block_sums[starts[:, :-1]].sum(axis=1)
            counts += full_block_counts[starts[:, :-1]].sum(axis=1)
        estimates[start_index:stop_index] = np.divide(
            totals, counts, out=np.full_like(totals, np.nan), where=counts > 0
        )
    estimates = estimates[np.isfinite(estimates)]
    if estimates.size < 2:
        raise ValueError("Too few nonempty bootstrap samples")
    lower, upper = np.quantile(estimates, [0.025, 0.975])
    return {
        "mean_difference": float(difference[valid].mean()),
        "ci95_lower": float(lower),
        "ci95_upper": float(upper),
    }


def diebold_mariano(
    loss_a: np.ndarray,
    loss_b: np.ndarray,
    *,
    horizon: int,
    newey_west_lag: int | None = None,
) -> dict[str, float]:
    difference = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    valid = np.isfinite(difference)
    n = int(valid.sum())
    if n < 3:
        return {"statistic": float("nan"), "p_value": float("nan")}
    lag = max(horizon - 1, 0) if newey_west_lag is None else newey_west_lag
    lag = min(lag, len(difference) - 1)
    if lag < 0:
        raise ValueError("newey_west_lag must be nonnegative")
    mean = difference[valid].mean()
    # HAC sandwich for the observed-loss mean on a regular hourly grid.
    # Missing slots have zero influence, not a shortened time separation.
    centered = np.where(valid, difference - mean, 0.0)
    variance_sum = np.dot(centered, centered)
    for k in range(1, lag + 1):
        weight = 1.0 - k / (lag + 1)
        variance_sum += 2.0 * weight * np.dot(centered[k:], centered[:-k])
    standard_error = np.sqrt(max(variance_sum, 1e-12) / n**2)
    statistic = mean / standard_error
    p_value = 2.0 * stats.t.sf(abs(statistic), df=n - 1)
    return {"statistic": float(statistic), "p_value": float(p_value)}


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, float)
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values)
    running = 0.0
    count = len(p_values)
    for rank, index in enumerate(order):
        value = min(1.0, (count - rank) * p_values[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted


def paired_wilcoxon(loss_a: np.ndarray, loss_b: np.ndarray) -> dict[str, float]:
    """Two-sided paired Wilcoxon test for pre-aggregated temporal blocks."""
    difference = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    difference = difference[np.isfinite(difference)]
    if difference.size < 2 or np.allclose(difference, 0.0):
        return {"statistic": float("nan"), "p_value": 1.0}
    result = stats.wilcoxon(difference, alternative="two-sided", method="auto")
    return {"statistic": float(result.statistic), "p_value": float(result.pvalue)}
