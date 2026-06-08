"""Figure 2 (+ Table 2) — environmental envelope (T vs NPP) with the six stations.

Mean temperature and NPP at each station placed within the global distribution of
ocean grid cells (1998-2019 time means). Prints Table 2 (per-station mean + IQR).

Inputs : data/forcings_global.zarr, data/stations_coords.json
Output : rehearsal/figures/Figure_2.png
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "rehearsal" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

with open(DATA_DIR / "stations_coords.json") as f:
    STATIONS = json.load(f)

COLORS = ["#332288", "#88CCEE", "#44AA99", "#117733", "#DDCC77", "#CC6677"]  # Tol palette
MARKERS = ["s", "^", "D", "v", "o", "p"]

# --- global climatology ----------------------------------------------------
forcings = xr.open_zarr(DATA_DIR / "forcings_global.zarr")
temperature, npp = forcings["temperature"], forcings["npp"]
temp_flat = temperature.mean(dim="T").compute().values.flatten()
npp_flat = npp.mean(dim="T").compute().values.flatten()
valid = ~(np.isnan(temp_flat) | np.isnan(npp_flat))
temp_clean, npp_clean = temp_flat[valid], npp_flat[valid]

# --- per-station statistics (Table 2) --------------------------------------
station_stats, rows = {}, []
for sid, info in STATIONS.items():
    lat_idx = int(np.abs(temperature["Y"].values - info["lat"]).argmin())
    lon_idx = int(np.abs(temperature["X"].values - info["lon"]).argmin())
    t_series = temperature.isel(Y=lat_idx, X=lon_idx).values
    p_series = npp.isel(Y=lat_idx, X=lon_idx).values
    m = ~(np.isnan(t_series) | np.isnan(p_series))
    t_valid, p_valid = t_series[m], p_series[m]
    stats = {
        "name": info["label"], "lat": info["lat"], "lon": info["lon"],
        "temp_mean": float(np.mean(t_valid)), "temp_q25": float(np.percentile(t_valid, 25)),
        "temp_q75": float(np.percentile(t_valid, 75)),
        "npp_mean": float(np.mean(p_valid)), "npp_q25": float(np.percentile(p_valid, 25)),
        "npp_q75": float(np.percentile(p_valid, 75)),
    }
    station_stats[sid] = stats
    rows.append(stats)

table_2 = pd.DataFrame(rows).set_index("name")[
    ["lat", "lon", "temp_mean", "temp_q25", "temp_q75", "npp_mean", "npp_q25", "npp_q75"]
].round(2)
print("Table 2:")
print(table_2.to_string())

# --- figure ----------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.5, 5.5), dpi=200)
x_min, x_max = float(temp_clean.min()) - 1, float(temp_clean.max()) + 1
y_min, y_max = float(npp_clean.min()) * 0.95, 1000.0
bins = [50, 30]

counts, _, _ = np.histogram2d(temp_clean, npp_clean, bins=bins, range=[[x_min, x_max], [y_min, y_max]])
vmax = float(np.percentile(counts[counts > 0], 95))
heatmap = ax.hist2d(temp_clean, npp_clean, bins=bins, range=[[x_min, x_max], [y_min, y_max]],
                    cmap="viridis", cmin=0, vmax=vmax, alpha=0.8)
cbar = plt.colorbar(heatmap[3], ax=ax, orientation="horizontal", pad=0.15, aspect=30, extend="max")
cbar.set_label("Number of grid points", fontweight="bold")

for i, s in enumerate(station_stats.values()):
    ax.errorbar(
        s["temp_mean"], s["npp_mean"],
        xerr=[[s["temp_mean"] - s["temp_q25"]], [s["temp_q75"] - s["temp_mean"]]],
        yerr=[[s["npp_mean"] - s["npp_q25"]], [s["npp_q75"] - s["npp_mean"]]],
        fmt=MARKERS[i], color=COLORS[i], markersize=10, capsize=4, capthick=1.5,
        elinewidth=1.5, ecolor="black", markeredgecolor="black", markeredgewidth=1.0, zorder=5,
    )

legend_handles = [
    Line2D([0], [0], marker=MARKERS[i], color="w", markerfacecolor=COLORS[i],
           markeredgecolor="black", markeredgewidth=1.0, markersize=8, label=s["name"], linestyle="None")
    for i, s in enumerate(station_stats.values())
]
legend_handles.append(Rectangle((0, 0), 1, 1, fc="none", ec="black", linewidth=1.5, label="IQR (Q25-Q75)"))

ax.set_xlabel("Mean temperature (°C)", fontweight="bold")
ax.set_ylabel(r"Mean primary production (mg C m$^{-2}$ d$^{-1}$)", fontweight="bold")
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.grid(axis="y", linestyle="--", alpha=0.3, linewidth=0.5)
ax.set_axisbelow(True)
ax.legend(handles=legend_handles, loc="upper right", framealpha=0.95, fontsize=9, ncol=1)

plt.tight_layout()
out = FIG_DIR / "Figure_2.png"
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved {out}")
