"""Build data/hot_real_observations.zarr from the HOT in-situ observations.

Same format/units as data/pseudo_observations.zarr (observed_biomass: T, Y, X;
g/m^2), but containing the REAL HOT zooplankton biomass at the sample days that
fall within the model window (2000-2019), at HOT's latitude (Y=23). This lets
run_optimization.py optimise against real observations via --obs-file.

Biomass recipe (same as experiment/02_hot_bats_observations.py / Figure R2):
  biomass_dry (mg DW/m^3) * 0.4 (C:DW) * layer_depth_surface (m) * 1e-3 -> g C/m^2,
  epipelagic_only, day/night merged (mean).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def _project_root(marker: str = "pyproject.toml") -> Path:
    for p in [Path.cwd(), *Path.cwd().parents]:
        if (p / marker).exists():
            return p
    raise FileNotFoundError(marker)


ROOT = _project_root()
DATA = ROOT / "data"
HOT = Path("/Users/ash/Documents/Workspace/SeapoPym-Data/src/hot/release/hot_zooplankton_obs.nc")
T_START, T_END = "2000-01-01", "2020-01-01"
HOT_LAT = 23.0

ds = xr.open_dataset(HOT).sel(depth_category="epipelagic_only")
bc = (ds.biomass_dry * 0.4 * ds.layer_depth_surface * 1e-3).mean("day_night")
df = bc.to_dataframe(name="b").reset_index().dropna(subset=["b"])
df = df[(df.time >= pd.Timestamp(T_START)) & (df.time < pd.Timestamp(T_END))]
df = df[df.b > 0]  # drop a single non-physical zero value
df["day"] = pd.to_datetime(df.time).dt.floor("D")
s = df.groupby("day")["b"].mean().sort_index()

T = s.index.values.astype("datetime64[ns]")
arr = s.values.astype("float32").reshape(len(T), 1, 1)
da = xr.DataArray(
    arr,
    dims=("T", "Y", "X"),
    coords={
        "T": ("T", T),
        "Y": ("Y", np.array([HOT_LAT], dtype="float32")),
        "X": ("X", np.array([0.0], dtype="float32")),
    },
    name="observed_biomass",
)
da["T"].attrs = {"axis": "T", "standard_name": "time"}
da["Y"].attrs = {"axis": "Y", "standard_name": "latitude", "units": "degrees_north"}
da["X"].attrs = {"axis": "X", "standard_name": "longitude", "units": "degrees_east"}
da.attrs["units"] = "gram / meter ** 2"

out = DATA / "hot_real_observations.zarr"
da.to_dataset().to_zarr(out, mode="w")
print(f"wrote {out}")
print(f"  {len(T)} obs days in {T_START}..{T_END} | median {float(s.median()):.3f} | std {float(s.std()):.3f} g/m^2")
