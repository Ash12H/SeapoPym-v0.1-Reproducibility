"""fig_equifinality.py — REHEARSAL equifinality figure (target vs best-fit).

For each station, overlay:
  - target  : the synthetic pseudo-observation (twin-experiment ground truth);
  - single  : biomass of the best individual of that station's optimisation;
  - MERGED  : biomass of the best individual of the joint (6-station) optimisation.

Goal: show that despite regime-dependent *parameter* recovery (e.g. tr_0 / gamma_tr
railing at warm stations), the *biomass* solution is nearly identical to the target
=> equifinality / non-identifiability (many parameter sets, same biomass). Directly
supports RC-3 #22 (the "circular" concern) and the sensitivity discussion.

Uses the production-config logbooks in rehearsal/logbooks/ (SBX, pop 256, NRMSE).
Re-runs SeapoPym with the GA's EXACT setup (Light model + precomputed IC + KernelParameter).

Output: rehearsal/figures/fig_equifinality.png + printed NRMSE table.
"""
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import yaml
from seapopym.configuration.no_transport import (
    ForcingParameter, ForcingUnit, FunctionalGroupParameter, FunctionalGroupUnit,
    FunctionalTypeParameter, KernelParameter, MigratoryTypeParameter, NoTransportConfiguration,
)
from seapopym.model.no_transport_model import NoTransportSpaceOptimizedLightModel

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data"
LOGS = HERE / "logbooks"
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)
WF = ("Weighted_fitness", "Weighted_fitness")

with open(ROOT / "parameters.yaml") as f:
    GS = yaml.safe_load(f)["global_simulation"]
T_START = GS["start_date"]
T_ANALYSIS_START = (datetime.strptime(GS["spin_up_end"], "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
T_END = "2019-12-31"
ZOOM = slice("2015-01-01", "2019-12-31")

# station id (logbook name) -> (latitude, display label), low to high latitude
STATIONS = [("HOT", 23.0, "HOT"), ("Canaries", 30.0, "CANARY"), ("BATS", 32.0, "BATS"),
            ("Bay_of_Biscay", 45.5, "BISCAY"), ("PAPA", 50.0, "PAPA"), ("BARENTS", 75.0, "BARENTS")]

raw = xr.open_zarr(DATA / "stations.zarr").sel(time=slice(T_START, T_END)).load()
ordered = raw.isel(station=raw.station_lat.argsort().values)
y_values = ordered.station_lat.values.astype(np.float32)
x_values = np.array([0.0], dtype=np.float32)


def _to_grid(da):
    arr = da.values.T[:, :, np.newaxis]
    return xr.DataArray(arr, dims=("T", "Y", "X"),
                        coords={"T": ordered.time.values, "Y": ("Y", y_values), "X": ("X", x_values)})


temperature = _to_grid(ordered.temperature).expand_dims(Z=[0], axis=1)
npp = _to_grid(ordered.npp)
for coord, axis in {"T": "T", "Z": "Z", "Y": "Y", "X": "X"}.items():
    if coord in temperature.coords:
        temperature[coord].attrs["axis"] = axis
    if coord in npp.coords:
        npp[coord].attrs["axis"] = axis
temperature.attrs["units"] = "degC"
npp.attrs["units"] = "mg/m2/day"
ic = xr.open_zarr(DATA / "initial_conditions.zarr").load()


def best_params(sid):
    df = pd.read_parquet(LOGS / f"ga_logbook_{sid}.parquet")
    return df.loc[df[WF].idxmax(), "Parametre"].to_dict()


def run_model(p):
    fg = FunctionalGroupParameter(functional_group=[FunctionalGroupUnit(
        name="zooplankton", energy_transfert=p["energy_transfert"],
        functional_type=FunctionalTypeParameter(
            lambda_temperature_0=p["lambda_temperature_0"], gamma_lambda_temperature=p["gamma_lambda_temperature"],
            tr_0=p["tr_0"], gamma_tr=p["gamma_tr"]),
        migratory_type=MigratoryTypeParameter(day_layer=0, night_layer=0))])
    config = NoTransportConfiguration(
        forcing=ForcingParameter(
            temperature=ForcingUnit(forcing=temperature), primary_production=ForcingUnit(forcing=npp),
            initial_condition_biomass=ForcingUnit(forcing=ic.biomass),
            initial_condition_production=ForcingUnit(forcing=ic.preproduction)),
        functional_group=fg, kernel=KernelParameter(biomass_solver="implicit"))
    with NoTransportSpaceOptimizedLightModel.from_configuration(configuration=config) as model:
        model.run()
        model.state.compute()
        return model.state.biomass.sel(T=slice(T_ANALYSIS_START, T_END), functional_group=0, drop=True).load()


def nrmse(pred, obs):
    pred, obs = np.asarray(pred, float), np.asarray(obs, float)
    m = np.isfinite(pred) & np.isfinite(obs)
    return float(np.sqrt(np.mean((pred[m] - obs[m]) ** 2)) / np.std(obs[m]))


target = xr.open_zarr(DATA / "pseudo_observations.zarr")["observed_biomass"]
print("Running MERGED best ...", flush=True)
merged_b = run_model(best_params("MERGED"))
print("Running single-station bests ...", flush=True)
single_b = {sid: run_model(best_params(sid)) for sid, _, _ in STATIONS}

fig, axes = plt.subplots(3, 2, figsize=(15, 10), sharex=True)
print(f"\n{'station':10s}{'NRMSE single':>14s}{'NRMSE MERGED':>14s}")
for ax, (sid, lat, label) in zip(axes.flat, STATIONS):
    tgt = target.sel(Y=lat, X=0, method="nearest")
    sgl = single_b[sid].sel(Y=lat, X=0, method="nearest")
    mrg = merged_b.sel(Y=lat, X=0, method="nearest")
    n_s, n_m = nrmse(sgl.values, tgt.values), nrmse(mrg.values, tgt.values)
    print(f"{label:10s}{n_s:14.4f}{n_m:14.4f}")
    tz, sz, mz = tgt.sel(T=ZOOM), sgl.sel(T=ZOOM), mrg.sel(T=ZOOM)
    ax.plot(tz["T"].values, tz.values, color="black", lw=2.0, label="target (pseudo-obs)", zorder=3)
    ax.plot(sz["T"].values, sz.values, color="#d62728", lw=1.2, label=f"single best (NRMSE={n_s:.3f})")
    ax.plot(mz["T"].values, mz.values, color="#1f77b4", lw=1.2, ls="--", label=f"MERGED best (NRMSE={n_m:.3f})")
    ax.set_title(f"{label} ({lat:g}°N)", fontweight="bold", fontsize=11)
    ax.set_ylabel(r"Biomass (g m$^{-2}$)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

fig.suptitle("Target vs best-fit biomass (single-station and MERGED) — SBX/pop256 — zoom 2015-2019",
             fontweight="bold", fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.97])
out = FIG / "Figure_9.png"
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
print(f"\nSaved {out}")
