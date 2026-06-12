"""Extract the processed in-situ zooplankton observations (HOT, BATS) from SeapoPym-Data into data/.

The SeapoPym-Data release files hold dry-mass zooplankton tows; here we keep the epipelagic samples,
convert to carbon areal biomass (g C m-2), and take the DAILY MEAN (average over day + night tows; no
diel or functional-group distinction — SEAPODYM-LMTL is total zooplankton):
    biomass_gC_m2 = mean_dn( biomass_dry * 0.4 (C:dry ratio) * layer_depth_surface * 1e-3 )
We use a uniform literature C:dry factor of 0.4 (not the file's biomass_carbon: BATS has none, and HOT's
uses ~0.35) so both stations are treated identically. The result is committed to data/ so reviewers can
access the observations behind Figure 5 WITHOUT the external SeapoPym-Data repo. (Source path is the
author's machine; re-run only to refresh.)

Source : <SeapoPym-Data>/src/{hot,bats}/release/{hot,bats}_zooplankton_obs.nc   (external)
Output : data/insitu_zooplankton_obs.csv   (columns: station, time, biomass_gC_m2)
Run    : .venv/bin/python scripts/data/extract_insitu_observations.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import xarray as xr

from seapopym_repro import paths

SEAPOPYM_DATA = Path("/Users/ash/Documents/Workspace/SeapoPym-Data/src")
STATIONS = {"HOT": "hot", "BATS": "bats"}   # display name -> SeapoPym-Data slug

frames = []
for name, slug in STATIONS.items():
    ds = xr.open_dataset(SEAPOPYM_DATA / slug / "release" / f"{slug}_zooplankton_obs.nc").sel(
        depth_category="epipelagic_only")
    # carbon areal biomass per tow -> DAILY MEAN over day+night (skipna: a day with one tow uses it)
    daily = (ds.biomass_dry * 0.4 * ds.layer_depth_surface * 1e-3).mean("day_night", skipna=True)
    b = daily.to_dataframe(name="biomass_gC_m2").reset_index().dropna(subset=["biomass_gC_m2"])
    b["station"] = name
    frames.append(b[["station", "time", "biomass_gC_m2"]])

df = pd.concat(frames, ignore_index=True).sort_values(["station", "time"])
out = paths.DATA / "insitu_zooplankton_obs.csv"
df.to_csv(out, index=False)
print(f"extracted {len(df)} epipelagic obs samples -> {out}  (per station: {df.station.value_counts().to_dict()})")
