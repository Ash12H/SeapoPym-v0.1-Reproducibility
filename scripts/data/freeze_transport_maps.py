"""Freeze the transport-impact comparison maps — the frozen product behind Figure 4.

Compares SeapoPym (0D, no transport; its global run = data/biomass_global.zarr) to the SEAPODYM-LMTL
operational reference (2D, with transport; the `zooc` field of data/forcings_global.zarr) over the
2000-2019 analysis window, reduced to four global maps:
    pred_mean  SeapoPym mean biomass                                  (g m-2)
    bias       mean(SeapoPym - LMTL)                                  (g m-2, signed)
    rmse       sqrt(mean((SeapoPym - LMTL)^2))                        (g m-2)
    mape       mean(|SeapoPym - LMTL| / |LMTL|) * 100                 (%)
Written as a small committed NetCDF so Figure 4 redraws from git without the heavy global zarr.

Inputs : data/biomass_global.zarr ('biomass'), data/forcings_global.zarr ('zooc')   (gitignored)
Output : products/transport_impact_maps.nc
Run    : .venv/bin/python scripts/data/freeze_transport_maps.py
"""
from __future__ import annotations

from datetime import datetime, timedelta

import cf_xarray.units  # noqa: F401  (registers CF unit conversions on pint)
import numpy as np
import pint_xarray  # noqa: F401  (xarray .pint accessor)
import xarray as xr

from seapopym_repro import paths

GS = paths.load_params()["global_simulation"]
T0 = (datetime.strptime(GS["spin_up_end"], "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
T1 = GS["end_date"]
sl = slice(T0, T1)

pred = (xr.open_zarr(paths.DATA / "biomass_global.zarr")["biomass"]
        .sel(T=sl, functional_group=0, drop=True).pint.quantify().pint.to("g/m^2").pint.dequantify())
obs = (xr.open_zarr(paths.DATA / "forcings_global.zarr")["zooc"]
       .sel(T=sl).pint.quantify().pint.to("g/m^2").pint.dequantify())

ocean = ~obs.isel(T=0).isnull()
pred = pred.where(ocean)
obs = obs.where(ocean)

maps = xr.Dataset({
    "pred_mean": pred.mean("T", skipna=False),
    "bias": (pred - obs).mean("T", skipna=False),
    "rmse": np.sqrt(((pred - obs) ** 2).mean("T", skipna=False)),
    "mape": (np.abs(pred - obs) / np.abs(obs)).where(obs > 0).mean("T", skipna=True) * 100.0,
}).compute()
maps.attrs["analysis_period"] = f"{T0}/{T1}"
maps.attrs["comparison"] = "SeapoPym 0D (biomass_global) vs SEAPODYM-LMTL 2D (forcings_global zooc)"

paths.PRODUCTS.mkdir(parents=True, exist_ok=True)
out = paths.PRODUCTS / "transport_impact_maps.nc"
maps.to_netcdf(out)
print(f"froze transport maps -> {out}  dims={dict(maps.sizes)}")
