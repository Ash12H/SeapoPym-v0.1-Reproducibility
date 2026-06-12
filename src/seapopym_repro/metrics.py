"""Cost-function comparators for the metric diagnostic (Figure 7 follow-up).

Same signature as the framework's MetricProtocol -- (prediction, observation) -> float -- so they can
be used both to re-score the best members under alternative metrics and, optionally, as the objective
of a re-optimisation. The framework ships nrmse_std + rmse; we add the mean-normalised NRMSE (relative
error, balanced across stations) and MAE.

Why nrmse_mean: std-normalisation (the current cost) divides by std(obs), which inflates low-
variability stations (HOT: std~0.06 turns a 0.004 RMSE into a 0.07 NRMSE). Normalising by the mean
gives a relative error that weights stations by magnitude, not variability. In a noise-free twin
experiment a log transform adds no value (the true optimum is exactly 0), so it is intentionally absent.
"""
from __future__ import annotations

import numpy as np


def rmse_comparator(prediction, observation) -> float:
    """Root mean square error (absolute)."""
    return float(np.sqrt(np.mean((np.asarray(prediction) - np.asarray(observation)) ** 2)))


def nrmse_std_comparator(prediction, observation) -> float:
    """RMSE normalised by the standard deviation of the observation (the framework's current cost)."""
    return rmse_comparator(prediction, observation) / float(np.std(observation))


def nrmse_mean_comparator(prediction, observation) -> float:
    """RMSE normalised by the mean of the observation (relative error; balanced across stations)."""
    return rmse_comparator(prediction, observation) / float(np.mean(observation))


def mae_comparator(prediction, observation) -> float:
    """Mean absolute error (absolute)."""
    return float(np.mean(np.abs(np.asarray(prediction) - np.asarray(observation))))


COMPARATORS = {
    "nrmse_std": nrmse_std_comparator,
    "nrmse_mean": nrmse_mean_comparator,
    "rmse": rmse_comparator,
    "mae": mae_comparator,
}
