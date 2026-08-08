"""Cost comparators available to the twin experiments.

Each takes a prediction and an observation and returns a float, the signature the framework expects,
so any of them can serve as the objective of an optimization or re-score an existing run. The
framework provides the NRMSE normalized by the standard deviation and the plain RMSE; this module
adds the NRMSE normalized by the mean, used in the paper, and the MAE.

Normalizing by the standard deviation inflates the cost at stations whose biomass varies little: at
HOT a standard deviation near 0.06 turns an RMSE of 0.004 into an NRMSE of 0.07. Normalizing by the
mean gives a relative error that weights stations by magnitude instead.
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
    """Mean absolute error (absolute). Equals the CRPS of a deterministic point forecast (RC-3)."""
    return float(np.mean(np.abs(np.asarray(prediction) - np.asarray(observation))))


def nmae_comparator(prediction, observation) -> float:
    """MAE normalised by the mean of the observation (relative absolute error).

    Equals a deterministic CRPS (CRPS -> MAE) normalised per station: answers RC-3's scoring-rule
    request AND stays balanced across stations (relative weighting, no std-inflation). Recommended
    candidate to compare/optimise across stations.
    """
    return mae_comparator(prediction, observation) / float(np.mean(observation))


COMPARATORS = {
    "nrmse_std": nrmse_std_comparator,
    "nrmse_mean": nrmse_mean_comparator,
    "rmse": rmse_comparator,
    "mae": mae_comparator,
    "nmae": nmae_comparator,
}
