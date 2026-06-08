"""Figure 4 — 4-panel transport-impact comparison (SeapoPym 0D vs SEAPODYM-LMTL 2D).

Addresses RC1 (show absolute discrepancies) and RC3 (add a MAPE panel). SeapoPym
(0D, no transport, reference parameters) vs the SEAPODYM-LMTL operational reference
(2D, with transport), over the analysis window, as a 2x2 panel:

  (a) SeapoPym mean biomass
  (b) Bias  = SeapoPym - SEAPODYM-LMTL   (signed, diverging)
  (c) MAPE  vs SEAPODYM-LMTL             (relative error, %)
  (d) RMSE  vs SEAPODYM-LMTL             (absolute error, g m-2)

Inputs : data/biomass_global.zarr ('biomass'), data/forcings_global.zarr ('zooc')
Output : rehearsal/figures/Figure_4.png
"""

from datetime import datetime, timedelta
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cf_xarray.units  # noqa: F401  (registers CF unit conversions on pint)
import matplotlib.pyplot as plt
import numpy as np
import pint_xarray  # noqa: F401  (xarray .pint accessor)
import xarray as xr
import yaml

BIOMASS_VMAX = 3.1   # g m-2
RMSE_VMAX = 1.7      # g m-2
MAPE_VMAX = 100.0    # %

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "rehearsal" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

with open(ROOT / "parameters.yaml") as f:
    GS = yaml.safe_load(f)["global_simulation"]
T_ANALYSIS_START = (datetime.strptime(GS["spin_up_end"], "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
T_END = GS[GS["mode"]]["end_date"]
print(f"Mode: {GS['mode']} | analysis window: {T_ANALYSIS_START} -> {T_END}")

time_slice = slice(T_ANALYSIS_START, T_END)
pred = (
    xr.open_zarr(DATA_DIR / "biomass_global.zarr")["biomass"]
    .sel(T=time_slice, functional_group=0, drop=True)
    .pint.quantify().pint.to("g/m^2").pint.dequantify()
)
obs = (
    xr.open_zarr(DATA_DIR / "forcings_global.zarr")["zooc"]
    .sel(T=time_slice)
    .pint.quantify().pint.to("g/m^2").pint.dequantify()
)

ocean = ~obs.isel(T=0).isnull()
pred = pred.where(ocean)
obs = obs.where(ocean)

pred_mean = pred.mean("T", skipna=False).compute()
bias = (pred - obs).mean("T", skipna=False).compute()
rmse = np.sqrt(((pred - obs) ** 2).mean("T", skipna=False)).compute()
mape = ((np.abs(pred - obs) / np.abs(obs)).where(obs > 0).mean("T", skipna=True) * 100.0).compute()

bias_lim = float(np.nanpercentile(np.abs(bias.values), 98))
bias_lim = round(bias_lim, 2) if bias_lim > 0 else 1.0


def _med(da: xr.DataArray) -> float:
    v = da.values
    return float(np.nanmedian(v[np.isfinite(v)]))


print(f"Median RMSE = {_med(rmse):.3f} g/m^2 | Median MAPE = {_med(mape):.1f}% "
      f"| Mean bias = {float(np.nanmean(bias.values[np.isfinite(bias.values)])):+.3f} g/m^2 "
      f"| bias colour-limit (p98 |bias|) = ±{bias_lim:g} g/m^2")


def add_cyclic_point(da: xr.DataArray, lon_name: str = "X") -> xr.DataArray:
    extra = da.isel({lon_name: 0})
    out = xr.concat([da, extra], dim=lon_name)
    return out.assign_coords({lon_name: list(da[lon_name].values) + [180]})


PANELS = [
    {"data": pred_mean, "cmap": "cividis",  "vmin": 0,          "vmax": BIOMASS_VMAX,
     "label": r"Biomass (g m$^{-2}$)",            "title": "(a) SeapoPym mean biomass", "extend": "max"},
    {"data": bias,      "cmap": "RdBu_r",   "vmin": -bias_lim,  "vmax": bias_lim,
     "label": r"Bias (g m$^{-2}$)",               "title": "(b) Bias (SeapoPym $-$ SEAPODYM-LMTL)", "extend": "both"},
    {"data": mape,      "cmap": "magma_r",  "vmin": 0,          "vmax": MAPE_VMAX,
     "label": r"MAPE (%)",                        "title": "(c) MAPE vs SEAPODYM-LMTL", "extend": "max"},
    {"data": rmse,      "cmap": "RdYlGn_r", "vmin": 0,          "vmax": RMSE_VMAX,
     "label": r"RMSE (g m$^{-2}$)",               "title": "(d) RMSE vs SEAPODYM-LMTL", "extend": "max"},
]

fig, axes = plt.subplots(
    2, 2, figsize=(18, 11),
    subplot_kw={"projection": ccrs.EckertIV(central_longitude=-80)},
)
for ax, panel in zip(axes.flat, PANELS):
    ax.set_global()
    ax.add_feature(cfeature.LAND, facecolor="lightgray", edgecolor="none", zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="black", zorder=2)
    m = add_cyclic_point(panel["data"]).plot(
        ax=ax, transform=ccrs.PlateCarree(),
        cmap=panel["cmap"], vmin=panel["vmin"], vmax=panel["vmax"],
        add_colorbar=False, zorder=1, extend=panel["extend"],
    )
    ax.gridlines(
        draw_labels=True, linewidth=0.5, color="gray", alpha=0.5, linestyle="--",
        x_inline=False, y_inline=False,
        xlabel_style={"size": 8}, ylabel_style={"size": 8},
    )
    cbar = plt.colorbar(m, ax=ax, orientation="horizontal", pad=0.05, shrink=0.8, extend=panel["extend"])
    cbar.set_label(panel["label"], fontsize=10)
    ax.set_title(panel["title"], fontsize=12, fontweight="bold")

plt.tight_layout()
out = FIG_DIR / "Figure_4.png"
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved {out}")
