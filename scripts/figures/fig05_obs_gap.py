"""Figure 5 — HOT/BATS, model vs in-situ observations: transport gap << structural gap.

Two panels (HOT, BATS), linear y. Per panel: in-situ daily-mean zooplankton biomass (scatter, P5-P95
clipped), SEAPODYM-LMTL (2D, with transport), SeapoPym (0D, reference params), and dotted total-mean
lines (obs, LMTL). Message: the two model means nearly coincide (transport gap ~0) while the obs mean
sits apart (structural gap) -> 0D-2D error << LMTL-obs error, quantified by the printed RMSEs.

Reads ONLY committed inputs (turnkey, no SeapoPym-Data, no framework):
  - data/insitu_zooplankton_obs.csv      daily-mean in-situ HOT/BATS obs, COMMITTED (derived from the
                                         SeapoPym-Data release; dry mass x 0.4 -> g C m-2)
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
from matplotlib.ticker import FuncFormatter

from seapopym_repro import figstyle as fs, paths

# colour-blind-safe (distinguished also by plot type): neutral obs, blue LMTL, vermillion SeapoPym
C_OBS, C_LMTL, C_SEAPO = "0.35", fs.BLUE, fs.ORANGE
T0, T1 = "2000-01-01", "2020-01-01"
ST = {"HOT": ("HOT", 23.0), "BATS": ("BATS", 32.0)}   # display -> (station key, latitude)

obs_all = pd.read_csv(paths.DATA / "insitu_zooplankton_obs.csv", parse_dates=["time"])
stations = xr.open_zarr(paths.DATA / "stations.zarr").sel(time=slice(T0, T1)).load()
seapo = xr.open_zarr(paths.DATA / "pseudo_observations.zarr").sel(T=slice(T0, T1)).load()


def load_obs(name):
    df = obs_all[(obs_all.station == name) & (obs_all.time >= pd.Timestamp(T0)) & (obs_all.time < pd.Timestamp(T1))]
    p5, p95 = df.biomass_gC_m2.quantile([0.05, 0.95])
    return df[(df.biomass_gC_m2 >= p5) & (df.biomass_gC_m2 <= p95) & (df.biomass_gC_m2 > 0)]


def rmse(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.sqrt(np.mean((a[m] - b[m]) ** 2)))


def _log_label(v, _):
    """Plain number, but only on the 1/2/3/5 mantissa ticks (keeps a sub-decade log axis uncluttered)."""
    if v <= 0:
        return ""
    m = round(v / 10 ** np.floor(np.log10(v)))
    return f"{v:g}" if m in (1, 2, 3, 5) else ""


def place_mean_labels(ax, means, min_frac=0.07):
    """Print each mean value just outside the right frame, nudged apart vertically if too close
    (the two model means nearly coincide, so their labels would otherwise overlap). Works in log
    space when the y-axis is logarithmic."""
    log = ax.get_yscale() == "log"
    fwd = (lambda v: np.log10(v)) if log else (lambda v: v)
    inv = (lambda v: 10 ** v) if log else (lambda v: v)
    ymin, ymax = ax.get_ylim()
    gap = min_frac * (fwd(ymax) - fwd(ymin))
    order = sorted(range(len(means)), key=lambda i: means[i][0])
    ys = [fwd(means[i][0]) for i in order]
    for k in range(1, len(ys)):
        ys[k] = max(ys[k], ys[k - 1] + gap)   # push up to keep a minimum separation
    for k, i in enumerate(order):
        ax.text(1.008, inv(ys[k]), f"{means[i][0]:.2f}", transform=ax.get_yaxis_transform(),
                color=means[i][1], va="center", ha="left", fontweight="bold", clip_on=False)


fig, axes = plt.subplots(2, 1, figsize=(fs.WIDTH_FULL, 0.62 * fs.WIDTH_FULL), sharex=True)
print(f"{'station':8s}{'RMSE transport (0D-2D)':>24s}{'RMSE LMTL-obs':>18s}{'ratio':>8s}")
for ax, (disp, (name, lat)) in zip(axes, ST.items()):
    obs = load_obs(disp)
    lm = stations.zooc.sel(station=name).to_pandas()
    sp = seapo.observed_biomass.sel(Y=lat, method="nearest").isel(X=0).to_pandas()

    ax.scatter(obs.time, obs.biomass_gC_m2, color=C_OBS, s=12, alpha=0.5, edgecolors="none")
    ax.plot(lm.index, lm.values, color=C_LMTL, lw=1.1)
    ax.plot(sp.index, sp.values, color=C_SEAPO, lw=1.1, ls="--")
    ax.set_yscale("log")   # set before placing the mean labels (place_mean_labels is log-aware)

    o_m, l_m, s_m = obs.biomass_gC_m2.mean(), float(lm.mean()), float(sp.mean())   # total means
    for val, col in [(o_m, C_OBS), (l_m, C_LMTL), (s_m, C_SEAPO)]:
        ax.axhline(val, color=col, ls=":", lw=1.6, alpha=0.9)
    place_mean_labels(ax, [(o_m, C_OBS), (l_m, C_LMTL), (s_m, C_SEAPO)])   # right-frame values, anti-overlap

    ax.set_title(disp, color="black")
    ax.set_ylabel(r"Biomass (g C m$^{-2}$)")
    ax.set_xlim(pd.Timestamp(T0), pd.Timestamp(T1))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.yaxis.set_major_formatter(FuncFormatter(_log_label))   # plain numbers, only 1/2/3/5 ticks
    ax.yaxis.set_minor_formatter(FuncFormatter(_log_label))
    ax.grid(True, which="both", alpha=0.25)

    # quantification: transport gap (SeapoPym vs SEAPODYM-LMTL) vs structural gap (SEAPODYM-LMTL vs obs)
    lm_on_sp = lm.reindex(sp.index, method="nearest").values
    lm_at_obs = lm.reindex(pd.to_datetime(obs.time.values), method="nearest").values
    r_transport = rmse(sp.values, lm_on_sp)
    r_obs = rmse(lm_at_obs, obs.biomass_gC_m2.values)
    print(f"{disp:8s}{r_transport:24.3f}{r_obs:18.3f}{r_obs / r_transport:8.1f}")

axes[-1].set_xlabel("Year")
handles = [
    Line2D([0], [0], marker="o", color="none", markerfacecolor=C_OBS, markersize=6, label="In-situ observations"),
    Line2D([0], [0], color=C_OBS, ls=":", lw=1.6, label="obs mean"),
    Line2D([0], [0], color=C_LMTL, lw=1.5, label="SEAPODYM-LMTL (transport, reference)"),
    Line2D([0], [0], color=C_LMTL, ls=":", lw=1.6, label="SEAPODYM-LMTL mean"),
    Line2D([0], [0], color=C_SEAPO, lw=1.5, ls="--", label="SeapoPym (no-transport, reference)"),
    Line2D([0], [0], color=C_SEAPO, ls=":", lw=1.6, label="SeapoPym mean"),
]
fig.tight_layout()
fig.legend(handles=handles, loc="lower center", ncol=3, frameon=True, bbox_to_anchor=(0.5, -0.09))
fs.save(fig, "Figure_5", subdir=None)
