"""Build the per-station summary table of the appendix.

For each station: the mean temperature, the recruitment age at the mean and at the warmest point of
the forcing series, in days and in daily cohorts, the difference between the two models sampled from
the maps of Figure 4, and, at HOT and BATS, the gap between the reference and the observations with
its ratio to that difference. The definitions are those of Figures 4 and 5, so the numbers match.

Inputs : products/transport_impact_maps.nc   the model difference maps
         data/stations.zarr                  station temperature and the SEAPODYM-LMTL reference
         data/pseudo_observations.zarr       SeapoPym at the reference parameters
         data/insitu_zooplankton_obs.csv     in-situ observations, clipped as in Figure 5
Output : products/station_summary.csv, and the table and its LaTeX rows printed to stdout
Run    : .venv/bin/python scripts/tables/build_station_summary.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def _root(marker: str = "pyproject.toml") -> Path:
    for p in [Path.cwd(), *Path.cwd().parents]:
        if (p / marker).exists():
            return p
    raise FileNotFoundError(marker)


ROOT = _root()
DATA, PROD = ROOT / "data", ROOT / "products"
T0, T1 = "2000-01-01", "2020-01-01"          # analysis period, exactly as fig05_obs_gap.py
TAU_R0, GAMMA_TAU = 10.38, -0.11             # Table 1 reference values (recruitment age)

# display name, station key in the zarr stores, latitude, longitude (cold -> warm)
SPECS = [
    ("BARENTS", "BARENTS", 75.0, 40.0),
    ("PAPA", "PAPA", 50.0, -132.0),
    ("BISCAY", "Bay_of_Biscay", 45.5, -4.0),
    ("CANARY", "Canaries", 30.0, -13.0),
    ("BATS", "BATS", 32.0, -64.0),
    ("HOT", "HOT", 23.0, -158.0),
]
POS = {"BARENTS": r"75\textdegree N, 40\textdegree E", "PAPA": r"50\textdegree N, 132\textdegree W",
       "BISCAY": r"45.5\textdegree N, 4\textdegree W", "CANARY": r"30\textdegree N, 13\textdegree W",
       "BATS": r"32\textdegree N, 64\textdegree W", "HOT": r"23\textdegree N, 158\textdegree W"}

obs_all = pd.read_csv(DATA / "insitu_zooplankton_obs.csv", parse_dates=["time"])
stations = xr.open_zarr(DATA / "stations.zarr").sel(time=slice(T0, T1)).load()
seapo = xr.open_zarr(DATA / "pseudo_observations.zarr").sel(T=slice(T0, T1)).load()
maps = xr.open_dataset(PROD / "transport_impact_maps.nc")


def load_obs(name):  # P5-P95 clip + strictly positive, identical to fig05_obs_gap.py
    df = obs_all[(obs_all.station == name) & (obs_all.time >= pd.Timestamp(T0)) & (obs_all.time < pd.Timestamp(T1))]
    p5, p95 = df.biomass_gC_m2.quantile([0.05, 0.95])
    return df[(df.biomass_gC_m2 >= p5) & (df.biomass_gC_m2 <= p95) & (df.biomass_gC_m2 > 0)]


def rmse(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.sqrt(np.mean((a[m] - b[m]) ** 2)))


def tau_r(temp_celsius):
    tf = np.maximum(np.asarray(temp_celsius, float), 0.0)   # floor at T_ref = 0 C (Eq. A3)
    tbar = 273 * tf / (273 + tf)                            # normalised temperature
    return TAU_R0 * np.exp(GAMMA_TAU * tbar)


rows = []
for disp, key, lat, lon in SPECS:
    cell = maps.sel(Y=lat, X=lon, method="nearest")
    temp = stations["temperature"].sel(station=key).values
    tau = tau_r(temp)
    rec = dict(station=disp, position=POS[disp], T_mean=float(np.nanmean(temp)),
               tau_mean=float(np.nanmean(tau)), tau_min=float(np.nanmin(tau)),
               rmse_0D2D=float(cell.rmse), mape_0D2D=float(cell.mape),
               rmse_2Dobs=np.nan, ratio=np.nan)
    if disp in ("HOT", "BATS"):
        obs = load_obs(disp)
        lm = stations.zooc.sel(station=key).to_pandas()
        sp = seapo.observed_biomass.sel(Y=lat, method="nearest").isel(X=0).to_pandas()
        r_transport = rmse(sp.values, lm.reindex(sp.index, method="nearest").values)
        r_obs = rmse(lm.reindex(pd.to_datetime(obs.time.values), method="nearest").values, obs.biomass_gC_m2.values)
        rec["rmse_2Dobs"], rec["ratio"] = r_obs, r_obs / r_transport
    rows.append(rec)

df = pd.DataFrame(rows)
PROD.mkdir(exist_ok=True)
df.to_csv(PROD / "station_summary.csv", index=False)

pd.set_option("display.width", 160)
print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
print("\nLaTeX body rows:\n")
for r in rows:
    obs = "--" if np.isnan(r["rmse_2Dobs"]) else f"{r['rmse_2Dobs']:.3f}"
    rat = "--" if np.isnan(r["ratio"]) else f"{r['ratio']:.1f}"
    print(f"{r['station']:8} & {r['position']:34} & {r['T_mean']:.1f} & {r['tau_mean']:.2f} & "
          f"{r['tau_min']:.2f} & {r['rmse_0D2D']:.3f} & {r['mape_0D2D']:.1f} & {obs} & {rat}\\\\")
print(f"\nwrote {PROD / 'station_summary.csv'}")
