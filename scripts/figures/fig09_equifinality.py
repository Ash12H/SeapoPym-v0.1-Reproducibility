"""Figure 9 — equifinality: the restart biomass spread is negligible despite divergent parameters.

For each single station, the twin target (black dashed), the best-fit biomass (thick, station colour),
and the min-max band of ALL seeded restarts (shaded, same colour). The restarts spread widely in
recovered parameters (recruitment equifinality at the warm stations) yet their biomass band is a thin
ribbon hugging the target -> many parameter sets, one biomass (RC-3 #22). MERGED is not shown.

Reads ONLY products/equifinality_biomass{_metric}.nc (frozen by freeze_equifinality_biomass.py) for
the production cost (paths.PRODUCTION_METRIC); member 0 is the best seed of each station.
Output : figures/Figure_9.{pdf,png}
Run    : .venv/bin/python scripts/figures/fig09_equifinality.py [--metric nrmse_mean]
"""
from __future__ import annotations

import argparse

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import xarray as xr
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from seapopym_repro import figstyle as fs, paths

ap = argparse.ArgumentParser()
ap.add_argument("--metric", default=paths.PRODUCTION_METRIC, help="cost metric whose frozen product to plot")
METRIC = ap.parse_args().metric

ds = xr.open_dataset(paths.PRODUCTS / f"equifinality_biomass{paths.metric_tag(METRIC)}.nc")
STATIONS = fs.order(ds.station.values.tolist())   # cold -> warm; single stations only (no MERGED)
time = ds.time.values

fig, axes = plt.subplots(2, 3, figsize=(fs.WIDTH_FULL, 0.60 * fs.WIDTH_FULL), sharex=True)
for ax, sid in zip(axes.flat, STATIONS):
    b = ds.biomass.sel(station=sid)               # (member, time); member 0 = best
    col = fs.color(sid)
    ax.fill_between(time, b.min("member"), b.max("member"), color=col, alpha=0.28,
                    linewidth=0, zorder=2)                                  # restart range (min-max)
    ax.plot(time, b.isel(member=0), color=col, lw=1.5, zorder=4)           # best fit: thick, station colour
    ax.plot(time, ds.target.sel(station=sid), color="black", lw=1.3, ls="--", zorder=5)   # truth: black dashed
    ax.set_title(fs.label(sid))
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    for tl in ax.get_xticklabels():
        tl.set_rotation(45); tl.set_ha("right")
for ax in axes[:, 0]:
    ax.set_ylabel(r"biomass (g C m$^{-2}$)")

handles = [Line2D([0], [0], color="black", lw=1.3, ls="--", label="target (pseudo-obs)"),
           Line2D([0], [0], color="0.4", lw=1.5, label="best fit"),
           Patch(facecolor="0.5", alpha=0.28, edgecolor="none", label="restart range (min–max)")]
fig.tight_layout(rect=[0, 0.06, 1, 1])
fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0), frameon=True)
fs.save(fig, "Figure_9", subdir=None)
