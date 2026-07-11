"""
Statistical helper functions for Ising Model analysis.
"""

import numpy as np
from scipy.signal import correlate, correlation_lags

# Defines the expected data types for the unified IsingResult arrays
METRIC_SCHEMA = {
    "magnetization": np.float64,
    "energy_mean": np.float64,
    "susceptibility": np.float64,
    "heat_capacity": np.float64,
    "magnetization_err": np.float64,
    "energy_mean_err": np.float64,
    "Chi_err": np.float64,
    "Cv_err": np.float64,
    "autocorrelation_time": np.float64,
    
    "sample_window": np.int64,
    "thinning_stride": np.int64,
    "thinned_sample_count": np.int64,
}

def autocorr(series: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Compute the autocorrelation and integrated autocorrelation time.

    Parameters
    ----------
    series : np.ndarray
        A 1D array containing the measurement data for every Monte Carlo time step.

    Returns
    -------
    tuple[np.ndarray, float]
        - A array of normalized autocorrelation values for positive lags.
        - The estimated integrated autocorrelation time (tau_int).
    """
    values = np.asarray(series, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("series must be one-dimensional")

    y = values - values.mean()
    if np.allclose(y, 0.0):
        return np.ones(values.size, dtype=np.float64), 0.5

    raw = correlate(y, y, mode="full", method="fft")
    lags = correlation_lags(values.size, values.size, mode="full")
    acorr = raw[lags >= 0]
    acorr /= acorr[0]

    if acorr.size == 0:
        return acorr, 0.0

    stop_index = np.where(acorr[1:] <= 0.0)[0]
    cutoff = acorr.size if stop_index.size == 0 else stop_index[0] + 1
    tau_int = 0.5 + float(acorr[1:cutoff].sum())

    return acorr, tau_int

def jackknife_error(
        samples: np.ndarray, statistic, tau_int: float, max_blocks: int = 20
    ) -> float:
    """
    Estimate the standard error of a given statistic using Block Jackknife resampling.

    Parameters
    ----------
    samples : np.ndarray
        A 1D array of raw equilibrium measurements.
    statistic : callable
        A function mapping a 1D array of samples to a float value.
    tau_int : float
        The integrated autocorrelation time of the samples.
    max_blocks : int, default=20
        The maximum number of blocks to divide the data into.

    Returns
    -------
    float
        The jackknife-estimated standard error for the provided statistic.
    """
    values = np.asarray(samples, dtype=np.float64)
    sample_count = values.size
    if sample_count < 2:
        return 0.0

    block_length = max(1, int(np.ceil(tau_int)))
    block_count = max(2, min(max_blocks, sample_count // block_length))
    if block_count < 2:
        return 0.0

    blocks = np.array_split(np.arange(sample_count), block_count)
    estimates = np.empty(len(blocks), dtype=np.float64)

    filled = 0
    for block in blocks:
        if block.size == 0:
            continue
        start, end = int(block[0]), int(block[-1]) + 1
        resample = np.concatenate((values[:start], values[end:]))
        if resample.size == 0:
            return 0.0
        estimates[filled] = float(statistic(resample))
        filled += 1

    estimates = estimates[:filled]
    if estimates.size < 2:
        return 0.0

    mean_estimate = float(estimates.mean())
    variance = (estimates.size - 1) / estimates.size * np.sum((estimates - mean_estimate) ** 2)
    return float(np.sqrt(max(variance, 0.0)))