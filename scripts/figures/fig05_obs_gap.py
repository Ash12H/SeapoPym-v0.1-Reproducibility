"""Figure 5 — HOT/BATS, model vs in-situ observations: transport gap << structural gap.

Two panels (HOT, BATS), linear y. Per panel: in-situ daily-mean zooplankton biomass (scatter, P5-P95
clipped), SEAPODYM-LMTL (2D, with transport), SeapoPym (0D, reference params), and dotted total-mean
lines (obs, LMTL). Message: the two model means nearly coincide (transport gap ~0) while the obs mean
sits apart (structural gap) -> 0D-2D error << model-obs error, quantified by the printed RMSEs.

Reads ONLY committed inputs (turnkey, no SeapoPym-Data, no framework):
  - data/insitu_zooplankton_obs.csv      extracted daily-mean in-situ obs (scripts/data/extract_insitu_observations.py)
  - data/stations.zarr ('zooc')          SEAPODYM-LMTL (2D, with transport)
  - data/pseudo_observations.zarr        SeapoPym (0D, reference parameters)
Output : figures/Figure_5.{pdf,png}
Run    : .venv/bin/python scripts/figures/fig05_obs_gap.py
"""
from __future__ import annotations

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.lines import Line2D

from seapopym_repro import figstyle as fs, paths

# colour-blind-safe (distinguished also by plot type): neutral obs, blue LMTL, vermillion SeapoPym
C_OBS, C_LMTL, C_SEAPO = "0.35", "#0072B2", "#D55E00"
T0, T1 = "2000-01-01", "2020-01-01"
ST = {"HOT": ("HOT", 23.0), "BATS": ("BATS", 32.0)}   # display -> (station key, latitude)

obs_all = pd.read_csv(paths.DATA / "insitu_zooplankton_obs.csv", parse_dates=["time"])
stations = xr.open_zarr(paths.DATA / "stations.zarr").sel(time=slice(T0, T1)).load()
seapo = xr.open_zarr(paths.DATA / "pseudo_observations.zarr").sel(T=slice(T0, T1)).load()


def load_obs(name):
    df = obs_all[(obs_all.station == name) & (obs_all.time >= pd.Timestamp(T0)) & (obs_all.time < pd.Timestamp(T1))]
    p5, p95 = df.biomass_gC_m2.quantile([0.05, 0.95])
    return df[(df.biomass_gC_m2 >= p5) & (df.biomass_gC_m2 <= p95)]


def rmse(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.sqrt(np.mean((a[m] - b[m]) ** 2)))


fig, axes = plt.subplots(2, 1, figsize=(fs.WIDTH_FULL, 0.62 * fs.WIDTH_FULL), sharex=True)
print(f"{'station':8s}{'RMSE transport (0D-2D)':>24s}{'RMSE model-obs':>18s}{'ratio':>8s}")
for ax, (disp, (name, lat)) in zip(axes, ST.items()):
    obs = load_obs(disp)
    lm = stations.zooc.sel(station=name).to_pandas()
    sp = seapo.observed_biomass.sel(Y=lat, method="nearest").isel(X=0).to_pandas()

    ax.scatter(obs.time, obs.biomass_gC_m2, color=C_OBS, s=12, alpha=0.5, edgecolors="none",
               label="In-situ observations")
    ax.plot(lm.index, lm.values, color=C_LMTL, lw=1.1, label="SEAPODYM-LMTL (2D, transport)")
    ax.plot(sp.index, sp.values, color=C_SEAPO, lw=1.1, ls="--", label="SeapoPym (0D, reference)")

    o_m, l_m = obs.biomass_gC_m2.mean(), float(lm.mean())   # obs + SEAPODYM-LMTL total means
    ax.axhline(o_m, color=C_OBS, ls=":", lw=1.6, alpha=0.9)
    ax.axhline(l_m, color=C_LMTL, ls=":", lw=1.6, alpha=0.9)
    for val, col in [(o_m, C_OBS), (l_m, C_LMTL)]:   # mean printed just outside the right frame
        ax.text(1.008, val, f"{val:.2f}", transform=ax.get_yaxis_transform(),
                color=col, va="center", ha="left", fontweight="bold", clip_on=False)

    ax.set_title(f"{disp} ({lat:g}°N)", color=fs.color(name))
    ax.set_ylabel(r"Biomass (g C m$^{-2}$)")
    ax.set_xlim(pd.Timestamp(T0), pd.Timestamp(T1))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(True, alpha=0.25)

    # quantification: transport gap (0D vs 2D) vs structural gap (model vs obs)
    sp_at_obs = sp.reindex(pd.to_datetime(obs.time.values), method="nearest").values
    lm_on_sp = lm.reindex(sp.index, method="nearest").values
    r_transport = rmse(sp.values, lm_on_sp)
    r_obs = rmse(sp_at_obs, obs.biomass_gC_m2.values)
    print(f"{disp:8s}{r_transport:24.3f}{r_obs:18.3f}{r_obs / r_transport:8.1f}")

axes[-1].set_xlabel("Year")
handles = [
    Line2D([0], [0], marker="o", color="none", markerfacecolor=C_OBS, markersize=6, label="In-situ observations"),
    Line2D([0], [0], color=C_LMTL, lw=1.5, label="SEAPODYM-LMTL (2D, transport)"),
    Line2D([0], [0], color=C_SEAPO, lw=1.5, ls="--", label="SeapoPym (0D, reference)"),
    Line2D([0], [0], color=C_OBS, ls=":", lw=1.6, label="obs mean"),
    Line2D([0], [0], color=C_LMTL, ls=":", lw=1.6, label="SEAPODYM-LMTL mean"),
]
axes[0].legend(handles=handles, loc="upper right", ncol=2, frameon=True, framealpha=0.95,
               facecolor="white", edgecolor="0.7")
fig.tight_layout()
fs.save(fig, "Figure_5", subdir=None)
