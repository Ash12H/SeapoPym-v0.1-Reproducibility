"""Figure R2 — HOT/BATS comparison with in-situ observations.

Two stacked subplots (HOT, BATS) over the full observation range.
- Scatter : in-situ observations (epipelagic_only, day + night merged).
- Line    : SEAPODYM-LMTL (2D, with transport).
- Line    : SeapoPym       (0D, no transport, reference parameters).

All in g C / m^2.
  HOT, BATS : biomass_dry (mg DW/m^3) × 0.4 (C:DW) × layer_depth_surface (m) × 1e-3

Time window : 2000-01-01 → 2020-01-01 (matches SeapoPym analysis window).
Outlier handling : obs outside [P5, P95] per station are excluded for clarity.
"""

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr


def _project_root(marker: str = "pyproject.toml") -> Path:
    for parent in [Path.cwd(), *Path.cwd().parents]:
        if (parent / marker).exists():
            return parent
    raise FileNotFoundError(f"marker {marker!r} not found")


PROJECT_ROOT = _project_root()
DATA_DIR = PROJECT_ROOT / "data"
FIG_DIR = PROJECT_ROOT / "experiment" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

OBS_BASE = Path("/Users/ash/Documents/Workspace/SeapoPym-Data/src")
HOT_PATH = OBS_BASE / "hot" / "release" / "hot_zooplankton_obs.nc"
BATS_PATH = OBS_BASE / "bats" / "release" / "bats_zooplankton_obs.nc"

C_TO_DW = 0.4  # zooplankton carbon : dry weight (Postel et al. 2000)
T_START = "2000-01-01"
T_END = "2020-01-01"


def load_obs(path: Path):
    """Return a tidy DataFrame: time, day_night, biomass_C (g C / m^2)."""
    ds = xr.open_dataset(path).sel(depth_category="epipelagic_only")
    integrated = ds.biomass_dry * C_TO_DW * ds.layer_depth_surface * 1e-3
    df = integrated.to_dataframe(name="biomass_C").reset_index()
    df = df.dropna(subset=["biomass_C"])
    return df[(df.time >= pd.Timestamp(T_START)) & (df.time < pd.Timestamp(T_END))]


hot_obs = load_obs(HOT_PATH)
bats_obs = load_obs(BATS_PATH)


def clip_p5_p95(df):
    p5, p95 = df.biomass_C.quantile([0.05, 0.95])
    return df[(df.biomass_C >= p5) & (df.biomass_C <= p95)]


hot_obs_full, bats_obs_full = hot_obs, bats_obs
hot_obs = clip_p5_p95(hot_obs_full)
bats_obs = clip_p5_p95(bats_obs_full)
print(f"HOT  obs: {len(hot_obs)}/{len(hot_obs_full)} samples after [P5, P95] clip | range g C/m^2 [{hot_obs.biomass_C.min():.3f}, {hot_obs.biomass_C.max():.3f}] | median {hot_obs.biomass_C.median():.3f}")
print(f"BATS obs: {len(bats_obs)}/{len(bats_obs_full)} samples after [P5, P95] clip | range g C/m^2 [{bats_obs.biomass_C.min():.3f}, {bats_obs.biomass_C.max():.3f}] | median {bats_obs.biomass_C.median():.3f}")

stations = xr.open_zarr(DATA_DIR / "stations.zarr").sel(time=slice(T_START, T_END)).load()
seapo = xr.open_zarr(DATA_DIR / "pseudo_observations.zarr").sel(T=slice(T_START, T_END)).load()

LMTL = {
    "HOT": stations.zooc.sel(station="HOT").to_pandas(),
    "BATS": stations.zooc.sel(station="BATS").to_pandas(),
}
SEAPO = {
    "HOT": seapo.observed_biomass.sel(Y=23.0, method="nearest").isel(X=0).to_pandas(),
    "BATS": seapo.observed_biomass.sel(Y=32.0, method="nearest").isel(X=0).to_pandas(),
}
print(f"LMTL  HOT: range [{LMTL['HOT'].min():.3f}, {LMTL['HOT'].max():.3f}] g C/m^2")
print(f"LMTL  BATS: range [{LMTL['BATS'].min():.3f}, {LMTL['BATS'].max():.3f}] g C/m^2")

COLOR_OBS = "#2ca02c"
COLOR_LMTL = "#1f77b4"
COLOR_SEAPO = "#d62728"

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

for ax, station, obs_df in zip(axes, ["HOT", "BATS"], [hot_obs, bats_obs]):
    ax.scatter(
        obs_df.time, obs_df.biomass_C,
        color=COLOR_OBS, s=18, alpha=0.6, edgecolors="none",
        label="In-situ observations (day + night)",
    )
    ax.plot(
        LMTL[station].index, LMTL[station].values,
        color=COLOR_LMTL, linewidth=1.2,
        label="SEAPODYM-LMTL (2D, with transport)",
    )
    ax.plot(
        SEAPO[station].index, SEAPO[station].values,
        color=COLOR_SEAPO, linewidth=1.2, linestyle="--",
        label="SeapoPym (0D, no transport)",
    )
    ax.set_title(station, fontsize=12, fontweight="bold")
    ax.set_ylabel(r"Biomass (g C m$^{-2}$)")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_xlim(pd.Timestamp(T_START), pd.Timestamp(T_END))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

axes[-1].set_xlabel("Year")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles, labels,
    loc="upper center", ncol=3, frameon=False, fontsize=10,
    bbox_to_anchor=(0.5, 1.02),
)
plt.tight_layout(rect=[0, 0, 1, 0.97])

out_pdf = FIG_DIR / "Figure_R2.pdf"
out_png = FIG_DIR / "Figure_R2.png"
fig.savefig(out_pdf, bbox_inches="tight")
fig.savefig(out_png, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved {out_pdf}, {out_png}")
