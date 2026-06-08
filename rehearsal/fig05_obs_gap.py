"""fig_X_obs_gap.py — DRAFT Figure X: HOT/BATS, model vs obs, gap comparison.

2 panels (HOT, BATS), LINEAR y. Per panel:
  - in-situ obs (scatter, P5-P95 clipped),
  - SEAPODYM-LMTL (2D, with transport),
  - SeapoPym (0D, no transport, reference params),
  - 3 dotted horizontal lines = total mean of obs / LMTL / SeapoPym.

Message: the two model means nearly coincide (transport gap ~0) while the obs mean
sits clearly apart (structural gap) -> 0D-2D error << model-obs error. Quantified by
printed RMSE(SeapoPym,LMTL) [transport] vs RMSE(model,obs) [structural].
"""
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIG = ROOT / "rehearsal" / "figures"
OBS_BASE = Path("/Users/ash/Documents/Workspace/SeapoPym-Data/src")
C_OBS, C_LMTL, C_SEAPO = "#2ca02c", "#1f77b4", "#d62728"
T0, T1 = "2000-01-01", "2020-01-01"
ST = {"HOT": ("HOT", 23.0), "BATS": ("BATS", 32.0)}

stations = xr.open_zarr(DATA / "stations.zarr").sel(time=slice(T0, T1)).load()
seapo = xr.open_zarr(DATA / "pseudo_observations.zarr").sel(T=slice(T0, T1)).load()


def load_obs(s):
    ds = xr.open_dataset(OBS_BASE / s.lower() / "release" / f"{s.lower()}_zooplankton_obs.nc").sel(depth_category="epipelagic_only")
    df = (ds.biomass_dry * 0.4 * ds.layer_depth_surface * 1e-3).to_dataframe(name="b").reset_index().dropna(subset=["b"])
    df = df[(df.time >= pd.Timestamp(T0)) & (df.time < pd.Timestamp(T1))]
    p5, p95 = df.b.quantile([0.05, 0.95])
    return df[(df.b >= p5) & (df.b <= p95)]


def rmse(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.sqrt(np.mean((a[m] - b[m]) ** 2)))


fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
print(f"{'station':8s}{'RMSE transport (0D-2D)':>24s}{'RMSE model-obs':>18s}{'ratio':>8s}")
for ax, (disp, (name, lat)) in zip(axes, ST.items()):
    obs = load_obs(disp)
    lm = stations.zooc.sel(station=name).to_pandas()
    sp = seapo.observed_biomass.sel(Y=lat, method="nearest").isel(X=0).to_pandas()

    ax.scatter(obs.time, obs.b, color=C_OBS, s=14, alpha=0.5, edgecolors="none", label="In-situ observations")
    ax.plot(lm.index, lm.values, color=C_LMTL, lw=1.1, label="SEAPODYM-LMTL (2D, transport)")
    ax.plot(sp.index, sp.values, color=C_SEAPO, lw=1.1, ls="--", label="SeapoPym (0D, reference)")

    o_m, l_m = obs.b.mean(), lm.mean()  # obs + SEAPODYM-LMTL means only
    ax.axhline(o_m, color=C_OBS, ls=":", lw=2, alpha=0.9)
    ax.axhline(l_m, color=C_LMTL, ls=":", lw=2, alpha=0.9)
    # mean values printed just OUTSIDE the right frame
    for val, col in [(o_m, C_OBS), (l_m, C_LMTL)]:
        ax.text(1.008, val, f"{val:.2f}", transform=ax.get_yaxis_transform(),
                color=col, va="center", ha="left", fontsize=9, fontweight="bold", clip_on=False)

    ax.set_title(f"{disp} ({lat:g}°N)", fontsize=12, fontweight="bold")
    ax.set_ylabel(r"Biomass (g C m$^{-2}$)")
    ax.set_xlim(pd.Timestamp(T0), pd.Timestamp(T1))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(True, alpha=0.25)

    # quantification: transport gap vs structural gap
    sp_at_obs = sp.reindex(pd.to_datetime(obs.time.values), method="nearest").values
    lm_on_sp = lm.reindex(sp.index, method="nearest").values
    r_transport = rmse(sp.values, lm_on_sp)
    r_obs = rmse(sp_at_obs, obs.b.values)
    print(f"{disp:8s}{r_transport:24.3f}{r_obs:18.3f}{r_obs / r_transport:8.1f}")

axes[-1].set_xlabel("Year")
legend_handles = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor=C_OBS, markersize=7, label="In-situ observations"),
    Line2D([0], [0], color=C_LMTL, lw=1.5, label="SEAPODYM-LMTL (2D, transport)"),
    Line2D([0], [0], color=C_SEAPO, lw=1.5, ls="--", label="SeapoPym (0D, reference)"),
    Line2D([0], [0], color=C_OBS, ls=":", lw=2, label="obs mean"),
    Line2D([0], [0], color=C_LMTL, ls=":", lw=2, label="SEAPODYM-LMTL mean"),
]
axes[1].legend(handles=legend_handles, loc="upper right", fontsize=8, framealpha=0.9)
fig.suptitle("HOT / BATS — model vs in-situ observations (dotted = total mean)", fontsize=13, fontweight="bold")
plt.tight_layout()
out = FIG / "Figure_5.png"
fig.savefig(out, dpi=190, bbox_inches="tight", facecolor="white")
print("wrote", out)
