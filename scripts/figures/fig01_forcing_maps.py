"""Figure 1: global maps of the forcing and the reference biomass, averaged over 2000-2019.

Four panels: (a) the mean zooplankton biomass of the SEAPODYM-LMTL reference, (b) the mean
vertically integrated NPP, (c) the mean epipelagic temperature, (d) the mean current norm
sqrt(U^2 + V^2). Each panel is titled by its letter alone, the colour-bar label carrying the
meaning.

Input  : products/forcing_global_means.nc, frozen by scripts/data/freeze_forcing_means.py
Output : figures/Figure_1.pdf and .png
Run    : .venv/bin/python scripts/figures/fig01_forcing_maps.py
"""
from __future__ import annotations

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import xarray as xr

from seapopym_repro import figstyle as fs, paths

means = xr.open_dataset(paths.PRODUCTS / "forcing_global_means.nc")


def add_cyclic_point(da, lon_name="X"):
    extra = da.isel({lon_name: 0})
    out = xr.concat([da, extra], dim=lon_name)
    return out.assign_coords({lon_name: list(da[lon_name].values) + [180]})


PANELS = [
    {"key": "zooc",         "cmap": "cividis", "vmin": 0,    "vmax": 3.5,  "label": r"SEAPODYM-LMTL mean biomass (g C m$^{-2}$)", "panel": "(a)"},
    {"key": "npp",          "cmap": "Greens",  "vmin": 0,    "vmax": 1100, "label": r"Mean net primary production (mg C m$^{-2}$ d$^{-1}$)", "panel": "(b)"},
    {"key": "temperature",  "cmap": "Reds",    "vmin": 0,    "vmax": 30,   "label": "Mean temperature (°C)",                   "panel": "(c)"},
    {"key": "current_norm", "cmap": "plasma",  "vmin": 0,    "vmax": 0.35, "label": r"Mean current norm (m s$^{-1}$)",          "panel": "(d)"},
]

fig, axes = plt.subplots(2, 2, figsize=(fs.WIDTH_FULL, 0.70 * fs.WIDTH_FULL),
                         subplot_kw={"projection": ccrs.EckertIV(central_longitude=-80)})
for ax, panel in zip(axes.flat, PANELS):
    data = add_cyclic_point(means[panel["key"]])
    ax.set_global()
    ax.add_feature(cfeature.LAND, facecolor="lightgray", edgecolor="none", zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.4, edgecolor="black", zorder=2)
    m = data.plot(ax=ax, transform=ccrs.PlateCarree(), cmap=panel["cmap"], vmin=panel["vmin"],
                  vmax=panel["vmax"], add_colorbar=False, zorder=1, rasterized=True)
    ax.gridlines(linewidth=0.4, color="gray", alpha=0.4, linestyle="--")
    cbar = plt.colorbar(m, ax=ax, orientation="horizontal", pad=0.04, shrink=0.85,
                        extend="max" if panel["vmax"] is not None else "neither")
    cbar.set_label(panel["label"])
    ax.set_title(panel["panel"], loc="left", fontweight="bold")
fig.tight_layout()
fs.save(fig, "Figure_1", subdir=None)
