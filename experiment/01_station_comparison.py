"""Figure R1 — Station-level comparison for review response #1 (and #4).

Per station, overlay:
- SEAPODYM-LMTL biomass (CMEMS, 2D operational reference, with transport)
- SeapoPym biomass (0D, no transport, reference parameters)

Inputs : data/stations.zarr, data/pseudo_observations.zarr
Output : experiment/figures/Figure_R1.{pdf,png}
"""

from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import xarray as xr
import yaml


def _project_root(marker: str = "pyproject.toml") -> Path:
    for parent in [Path.cwd(), *Path.cwd().parents]:
        if (parent / marker).exists():
            return parent
    raise FileNotFoundError(f"Project root marker {marker!r} not found.")


PROJECT_ROOT = _project_root()
DATA_DIR = PROJECT_ROOT / "data"
FIG_DIR = PROJECT_ROOT / "experiment" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

with open(PROJECT_ROOT / "parameters.yaml") as f:
    PARAMS = yaml.safe_load(f)
GS = PARAMS["global_simulation"]
MODE = GS["mode"]
T_ANALYSIS_START = (
    datetime.strptime(GS["spin_up_end"], "%Y-%m-%d") + timedelta(days=1)
).strftime("%Y-%m-%d")
T_END = GS[MODE]["end_date"]
print(f"Mode: {MODE} | analysis window: {T_ANALYSIS_START} -> {T_END}")

# Stations from low to high latitude
STATIONS_BY_LAT = [
    ("HOT",           23.0),
    ("Canaries",      30.0),
    ("BATS",          32.0),
    ("Bay_of_Biscay", 45.5),
    ("PAPA",          50.0),
    ("BARENTS",       75.0),
]

stations = (
    xr.open_zarr(DATA_DIR / "stations.zarr")
    .sel(time=slice(T_ANALYSIS_START, T_END))
    .load()
)
seapo = (
    xr.open_zarr(DATA_DIR / "pseudo_observations.zarr")
    .sel(T=slice(T_ANALYSIS_START, T_END))
    .load()
)

COLOR_LMTL = "#1f77b4"
COLOR_SEAPO = "#d62728"

fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex=True)

for i, (name, lat) in enumerate(STATIONS_BY_LAT):
    ax = axes.flat[i]
    lmtl_ts = stations.zooc.sel(station=name)
    seapo_ts = seapo.observed_biomass.sel(Y=lat, method="nearest").isel(X=0)

    ax.plot(
        stations.time.values, lmtl_ts.values,
        label="SEAPODYM-LMTL (2D, with transport)",
        color=COLOR_LMTL, linewidth=1.2,
    )
    ax.plot(
        seapo.T.values, seapo_ts.values,
        label="SeapoPym (0D, no transport)",
        color=COLOR_SEAPO, linewidth=1.2, linestyle="--",
    )

    label = name.replace("_", " ")
    ax.set_title(f"{label}  ({lat:g}°N)", fontsize=11, fontweight="bold")
    if i % 3 == 0:
        ax.set_ylabel(r"Biomass (g m$^{-2}$)")
    if i >= 3:
        ax.set_xlabel("Year")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[1, 7]))
    ax.tick_params(axis="x", labelsize=9)

handles, labels = axes.flat[0].get_legend_handles_labels()
fig.legend(
    handles, labels,
    loc="upper center", ncol=2, frameon=False, fontsize=10,
    bbox_to_anchor=(0.5, 1.02),
)

plt.tight_layout(rect=[0, 0, 1, 0.97])
out_pdf = FIG_DIR / "Figure_R1.pdf"
out_png = FIG_DIR / "Figure_R1.png"
fig.savefig(out_pdf, bbox_inches="tight")
fig.savefig(out_png, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved {out_pdf}")
print(f"Saved {out_png}")
