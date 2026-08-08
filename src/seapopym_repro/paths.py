"""Paths of the deposit, resolved once for every script.

The repository root is found by walking up to the directory holding pyproject.toml, so the paths are
correct wherever a script is run from.

    data/         committed inputs: station forcing, synthetic observations, in-situ observations
    products/     committed experiment outputs, what the figures read
    figures/      produced figures, PDF and PNG
    results_raw/  heavy intermediates, not committed
"""
from __future__ import annotations

from pathlib import Path

import yaml


def _find_root(start: Path) -> Path:
    for p in (start, *start.parents):
        if (p / "pyproject.toml").is_file():
            return p
    raise RuntimeError(f"repo root (pyproject.toml) not found above {start}")


ROOT = _find_root(Path(__file__).resolve())
DATA = ROOT / "data"
PRODUCTS = ROOT / "products"
FIGURES = ROOT / "figures"
RESULTS_RAW = ROOT / "results_raw"
CONFIG = ROOT / "parameters.yaml"

# The cost used in the paper: the NRMSE normalized by the mean of the target series. The optimum of
# a twin experiment does not depend on this choice, since the reference parameters give a zero cost
# under any metric, but the normalization sets how stations are weighted against one another. The
# mean denominator avoids the inflation that the standard deviation produces at HOT, whose biomass
# varies little. Runs with another metric are kept under a suffixed name.
PRODUCTION_METRIC = "nrmse_mean"
CMAES_LAMBDA = 8   # CMA-ES population of the published ensemble; product names carry it as l{lambda}


def metric_tag(metric: str) -> str:
    """Filename suffix for a cost metric, empty for nrmse_std and "_{metric}" otherwise."""
    return "" if metric == "nrmse_std" else f"_{metric}"


def cmaes_product(kind: str, metric: str | None = None, lam: int | None = None) -> Path:
    """Path to a CMA-ES product, named after its population size and cost metric.

    kind   : "seed_ensemble" for the per-restart results, "convergence_traces" for the cost curves
    metric : cost metric, PRODUCTION_METRIC by default
    """
    metric = metric or PRODUCTION_METRIC
    lam = lam or CMAES_LAMBDA
    return PRODUCTS / f"cmaes_{kind}_l{lam}{metric_tag(metric)}.csv"


def load_params() -> dict:
    """Parse parameters.yaml (reference values, bounds, optimiser/run settings)."""
    with open(CONFIG) as f:
        return yaml.safe_load(f)
