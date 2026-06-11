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


def load_params() -> dict:
    """Parse parameters.yaml (reference values, bounds, optimiser/run settings)."""
    with open(CONFIG) as f:
        return yaml.safe_load(f)
