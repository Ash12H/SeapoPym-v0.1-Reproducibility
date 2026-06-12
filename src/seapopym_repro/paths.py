"""Canonical paths for the reproducibility deposit — resolved ONCE, no Path(__file__).parents[N]
scattered across scripts. The repo root is found by walking up to the directory holding
pyproject.toml, so paths are correct whether a script runs from the repo root or elsewhere.

Layout (see Review/figure-guidelines.md and README):
    data/         committed INPUTS (small): station forcing, pseudo-observations, coords, real obs
    products/     frozen experiment outputs the figures consume (committed CSV — the display contract)
    figures/      produced figures (PDF + PNG, committed)
    results_raw/  heavy intermediates (GITIGNORED): per-seed parquets, raw logbooks, global zarr
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

# ---- production cost metric -----------------------------------------------------------------
# The cost the paper's figures are built on. The twin's optimum is metric-independent (true params
# give 0 under any metric); the choice rebalances inter-station weighting and per-seed reliability.
# We use mean-normalised NRMSE (RMSE/mean): same RMSE numerator as the original NRMSE, but the mean
# denominator removes the std-normalisation that inflated low-variability stations (HOT). The
# alternative-metric runs (nrmse_std, nmae, ...) are kept as suffixed products for the diagnostic.
PRODUCTION_METRIC = "nrmse_mean"
CMAES_LAMBDA = 8   # CMA-ES population the ensemble was run at (product filenames are namespaced l{lambda})


def metric_tag(metric: str) -> str:
    """Filename suffix for a cost metric: "" for nrmse_std (byte-stable legacy), "_{metric}" otherwise."""
    return "" if metric == "nrmse_std" else f"_{metric}"


def cmaes_product(kind: str, metric: str | None = None, lam: int | None = None) -> Path:
    """Path to a frozen CMA-ES product, namespaced by lambda + cost metric.

    kind   : "seed_ensemble" (per (exp, seed) recovery) or "convergence_traces" (best-so-far curves)
    metric : cost metric (default PRODUCTION_METRIC); use "nrmse_std"/"nmae"/... for the comparators
    Centralising this here means one constant (PRODUCTION_METRIC) decides which run the figures read,
    and a metric-suffixed glob can never silently grab the wrong file.
    """
    metric = metric or PRODUCTION_METRIC
    lam = lam or CMAES_LAMBDA
    return PRODUCTS / f"cmaes_{kind}_l{lam}{metric_tag(metric)}.csv"


def load_params() -> dict:
    """Parse parameters.yaml (reference values, bounds, optimiser/run settings)."""
    with open(CONFIG) as f:
        return yaml.safe_load(f)
