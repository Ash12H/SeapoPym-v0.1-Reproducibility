"""Figure 4 — 4-panel transport-impact comparison (SeapoPym 0D vs SEAPODYM-LMTL 2D).

Addresses RC1 (show absolute discrepancies) and RC3 (add a MAPE panel). SeapoPym (0D, no transport,
reference parameters) vs the SEAPODYM-LMTL operational reference (2D, with transport), as a 2x2 panel:
  (a) SeapoPym mean biomass
  (b) RMSE  vs SEAPODYM-LMTL             (absolute error, g m-2)
  (c) MAPE  vs SEAPODYM-LMTL             (relative error, %)
  (d) Bias  = SeapoPym - SEAPODYM-LMTL   (signed, diverging)
Subplot titles are just the panel letter; the colour-bar label carries the full meaning.

Reads ONLY the committed product products/transport_impact_maps.nc (produced by
scripts/data/freeze_transport_maps.py) — no heavy global zarr needed at figure time.
Output : figures/Figure_4.{pdf,png}
Run    : .venv/bin/python scripts/figures/fig04_transport.py
"""
from __future__ import annotations

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from seapopym_repro import figstyle as fs, paths

BIOMASS_VMAX = 3.5   # g C m-2 (matches Figure 1a for a direct biomass comparison)
RMSE_VMAX = 1.7      # g C m-2
MAPE_VMAX = 100.0    # %

maps = xr.open_dataset(paths.PRODUCTS / "transport_impact_maps.nc")
bias_lim = round(float(np.nanpercentile(np.abs(maps["bias"].values), 98)), 2) or 1.0


def add_cyclic_point(da, lon_name="X"):
    extra = da.isel({lon_name: 0})
    out = xr.concat([da, extra], dim=lon_name)
    return out.assign_coords({lon_name: list(da[lon_name].values) + [180]})


PANELS = [
    {"key": "pred_mean", "cmap": "cividis", "vmin": 0,         "vmax": BIOMASS_VMAX, "extend": "max",
     "label": r"SeapoPym mean biomass (g C m$^{-2}$)",      "panel": "(a)"},
    {"key": "rmse",      "cmap": "Reds",    "vmin": 0,         "vmax": RMSE_VMAX,    "extend": "max",
     "label": r"Root mean square error (g C m$^{-2}$)",     "panel": "(b)"},
    {"key": "mape",      "cmap": "Reds",    "vmin": 0,         "vmax": MAPE_VMAX,    "extend": "max",
     "label": "Mean absolute percentage error (%)",         "panel": "(c)"},
    {"key": "bias",      "cmap": "RdBu_r",  "vmin": -bias_lim, "vmax": bias_lim,     "extend": "both",
     "label": r"Bias (g C m$^{-2}$)",                       "panel": "(d)"},
]

fig, axes = plt.subplots(2, 2, figsize=(fs.WIDTH_FULL, 0.70 * fs.WIDTH_FULL),
                         subplot_kw={"projection": ccrs.EckertIV(central_longitude=-80)})
for ax, panel in zip(axes.flat, PANELS):
    data = add_cyclic_point(maps[panel["key"]])
    ax.set_global()
    ax.add_feature(cfeature.LAND, facecolor="lightgray", edgecolor="none", zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.4, edgecolor="black", zorder=2)
    m = data.plot(ax=ax, transform=ccrs.PlateCarree(), cmap=panel["cmap"], vmin=panel["vmin"],
                  vmax=panel["vmax"], add_colorbar=False, zorder=1, extend=panel["extend"], rasterized=True)
    ax.gridlines(linewidth=0.4, color="gray", alpha=0.4, linestyle="--")
    cbar = plt.colorbar(m, ax=ax, orientation="horizontal", pad=0.04, shrink=0.85, extend=panel["extend"])
    cbar.set_label(panel["label"])
    ax.set_title(panel["panel"], loc="left", fontweight="bold")
fig.tight_layout()
fs.save(fig, "Figure_4", subdir=None)
