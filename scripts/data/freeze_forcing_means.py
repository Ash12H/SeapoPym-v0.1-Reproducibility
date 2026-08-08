"""Reduce the global forcing to the time-mean maps behind Figures 1 and 2.

Reduces the heavy global forcing (data/forcings_global.zarr, ~6 GB, gitignored) to the 2000-2019
time-mean of the four fields the maps need, on the 1-degree grid (~180x360). The result is a small
NetCDF committed to products/, so the figures redraw without the raw forcing, which is only needed
to rebuild this product.

Input  : data/forcings_global.zarr            (gitignored; fetched by download_cmems_global.py)
Output : products/forcing_global_means.nc      (temperature, npp, zooc, current_norm on (Y, X))
Run    : .venv/bin/python scripts/data/freeze_forcing_means.py
"""
from __future__ import annotations

import numpy as np
import xarray as xr

from seapopym_repro import paths

ANALYSIS_START, ANALYSIS_END = "2000-01-01", "2019-12-31"   # 2-year spin-up excluded (forcing starts 1998)
src = paths.DATA / "forcings_global.zarr"
out = paths.PRODUCTS / "forcing_global_means.nc"

ds = xr.open_zarr(src).sel(T=slice(ANALYSIS_START, ANALYSIS_END))
means = xr.Dataset({
    "temperature":  ds.temperature.mean("T", skipna=False),
    "npp":          ds.npp.mean("T", skipna=False),
    "zooc":         ds.zooc.mean("T", skipna=False),
    "current_norm": np.sqrt(ds.U ** 2 + ds.V ** 2).mean("T", skipna=False),
}).compute()
means.attrs["analysis_period"] = f"{ANALYSIS_START}/{ANALYSIS_END}"
means.attrs["source"] = "data/forcings_global.zarr (CMEMS GLOBAL_MULTIYEAR_BGC_001_033 + SEAPODYM-LMTL)"

paths.PRODUCTS.mkdir(parents=True, exist_ok=True)
means.to_netcdf(out)
print(f"froze forcing means -> {out}  dims={dict(means.sizes)}")
