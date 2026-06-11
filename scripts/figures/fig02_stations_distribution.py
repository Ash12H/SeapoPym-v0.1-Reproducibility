"""Figure 2 (+ Table 2) — environmental envelope (T vs NPP) with the six stations.

Each station's mean temperature and NPP (with IQR) placed within the global distribution of ocean
grid cells (2000-2019 time means). Prints Table 2 (per-station mean + IQR).

Reads ONLY committed inputs (turnkey, no raw global forcing):
  - products/forcing_global_means.nc   global per-pixel time-mean T + NPP (the cloud)
  - data/stations.zarr                 per-station T + NPP time series (the per-station mean + IQR)
Station colour + marker come from figstyle (same identity as every other figure).
Output : figures/Figure_2.{pdf,png}
Run    : .venv/bin/python scripts/figures/fig02_stations_distribution.py
"""
from __future__ import annotations

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde

from seapopym_repro import figstyle as fs, paths

A0, A1 = "2000-01-01", "2019-12-31"   # analysis period (2-year spin-up excluded), as in Figure 1

# --- global T-NPP cloud: per-pixel time means (frozen product) -------------------------------
means = xr.open_dataset(paths.PRODUCTS / "forcing_global_means.nc")
temp_flat, npp_flat = means.temperature.values.flatten(), means.npp.values.flatten()
valid = ~(np.isnan(temp_flat) | np.isnan(npp_flat))
temp_clean, npp_clean = temp_flat[valid], npp_flat[valid]

# --- per-station mean + IQR over the analysis period (committed station series) --------------
st = xr.open_zarr(paths.DATA / "stations.zarr").sel(time=slice(A0, A1))
STATIONS = fs.order(st["station"].values)      # the 6 single stations, coldest -> warmest
rows = {}
for sid in STATIONS:
    t = st.temperature.sel(station=sid).values
    p = st.npp.sel(station=sid).values
    m = ~(np.isnan(t) | np.isnan(p))
    t, p = t[m], p[m]
    rows[sid] = {
        "name": fs.label(sid), "lat": float(st.station_lat.sel(station=sid)),
        "temp_mean": float(np.mean(t)), "temp_q25": float(np.percentile(t, 25)), "temp_q75": float(np.percentile(t, 75)),
        "npp_mean": float(np.mean(p)), "npp_q25": float(np.percentile(p, 25)), "npp_q75": float(np.percentile(p, 75)),
    }

table_2 = pd.DataFrame(rows.values()).set_index("name").round(2)
print("Table 2 (analysis period 2000-2019):")
print(table_2.to_string())

# --- figure ----------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(0.72 * fs.WIDTH_FULL, 0.58 * fs.WIDTH_FULL))
x_min, x_max = float(temp_clean.min()) - 1, float(temp_clean.max()) + 1
y_min, y_max = 0.0, 1000.0

# density-coloured scatter: every ocean grid point, coloured by its local density (gaussian_kde,
# which whitens by the data covariance so the T vs NPP scale difference is handled). Dense points
# drawn last. Honest (all points shown) + smooth (no binning artefact). rasterized -> light PDF.
disp = (npp_clean >= y_min) & (npp_clean <= y_max)
xd, yd = temp_clean[disp], npp_clean[disp]
dens = gaussian_kde(np.vstack([xd, yd]))(np.vstack([xd, yd]))
dens /= dens.max()
order = dens.argsort()
sc = ax.scatter(xd[order], yd[order], c=dens[order], s=4, cmap="viridis",
                edgecolors="none", vmin=0, vmax=1, rasterized=True)
cbar = ax.figure.colorbar(sc, ax=ax, orientation="horizontal", pad=0.16, aspect=30)
cbar.set_label("Relative density of ocean grid points")

for sid in STATIONS:
    s = rows[sid]
    eb = ax.errorbar(
        s["temp_mean"], s["npp_mean"],
        xerr=[[s["temp_mean"] - s["temp_q25"]], [s["temp_q75"] - s["temp_mean"]]],
        yerr=[[s["npp_mean"] - s["npp_q25"]], [s["npp_q75"] - s["npp_mean"]]],
        fmt=fs.marker(sid), color=fs.color(sid), markersize=9, capsize=3, capthick=1.4,
        elinewidth=1.4, ecolor="black", markeredgecolor="black", markeredgewidth=0.8, zorder=6,
    )
    halo = [pe.withStroke(linewidth=2.8, foreground="white")]   # white halo -> legible over the scatter
    for ln in eb[1]:        # cap lines
        ln.set_path_effects(halo)
    for lc in eb[2]:        # bar line collections
        lc.set_path_effects(halo)

handles = [Line2D([0], [0], marker=fs.marker(sid), color="none", markerfacecolor=fs.color(sid),
                  markeredgecolor="black", markersize=8, linestyle="None", label=fs.label(sid))
           for sid in STATIONS]
handles.append(Line2D([0], [0], color="black", linewidth=1.4, marker="|", markersize=9,
                       markeredgewidth=1.4, label="IQR (Q25-Q75)"))

ax.set_xlabel("Mean temperature (°C)")
ax.set_ylabel(r"Mean primary production (mg C m$^{-2}$ d$^{-1}$)")
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.grid(axis="y", linestyle="--", alpha=0.3, linewidth=0.5)
ax.set_axisbelow(True)
# force a frame here (figstyle disables it globally): the legend sits over the scatter, so it needs
# an opaque white background to stay readable
leg = ax.legend(handles=handles, loc="upper right", ncol=1, frameon=True, framealpha=0.95,
                facecolor="white", edgecolor="0.7", fancybox=False)
leg.set_zorder(10)

fig.tight_layout()
fs.save(fig, "Figure_2", subdir=None)
