"""HOT real-data experiment — seasonal phase mismatch.

Monthly climatology (2000-2019) of:
  - in-situ HOT zooplankton observations (mean +/- 1 std of the monthly samples),
  - SeapoPym with reference parameters (= pseudo_observations at Y=23).

Shows that observed HOT zooplankton peaks in SUMMER while the NPP-driven model
peaks in WINTER: an out-of-phase seasonal cycle that parameter optimisation
cannot fix (the GA hit the parameter bounds at NRMSE ~ 1).
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def _project_root(marker: str = "pyproject.toml") -> Path:
    for p in [Path.cwd(), *Path.cwd().parents]:
        if (p / marker).exists():
            return p
    raise FileNotFoundError(marker)


ROOT = _project_root()
DATA = ROOT / "data"
FIG = ROOT / "experiment" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

obs = xr.open_zarr(DATA / "hot_real_observations.zarr")["observed_biomass"].isel(Y=0, X=0).to_pandas().dropna()
ref = xr.open_zarr(DATA / "pseudo_observations.zarr")["observed_biomass"].sel(Y=23.0, method="nearest").isel(X=0).to_pandas()

months = np.arange(1, 13)
obs_mean = obs.groupby(obs.index.month).mean().reindex(months)
obs_std = obs.groupby(obs.index.month).std().reindex(months)
ref_mean = ref.groupby(ref.index.month).mean().reindex(months)

fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
ax.fill_between(months, obs_mean - obs_std, obs_mean + obs_std, color="#2ca02c", alpha=0.18)
ax.plot(months, obs_mean, "o-", color="#2ca02c", lw=2, label="In-situ observations (HOT)")
ax.plot(months, ref_mean, "s--", color="#d62728", lw=2, label="SeapoPym (reference parameters)")
ax.set_xticks(months)
ax.set_xticklabels(["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
ax.set_xlabel("Month")
ax.set_ylabel(r"Zooplankton biomass (g C m$^{-2}$)")
ax.set_title("HOT — monthly climatology (2000–2019): observed summer peak vs modelled winter peak",
             fontsize=10, fontweight="bold")
ax.grid(True, alpha=0.3)
ax.legend(frameon=False)
plt.tight_layout()
fig.savefig(FIG / "HOT_real_climatology.png", dpi=200, bbox_inches="tight", facecolor="white")
print("wrote", FIG / "HOT_real_climatology.png")
