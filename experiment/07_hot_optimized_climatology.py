"""HOT climatology with the OPTIMISED-parameter run added.

Runs SeapoPym with the parameters found by the real-data optimisation
(ga_logbook_HOT_validation_hotreal.parquet), extracts HOT, and overlays the
monthly climatology of: in-situ obs, reference-parameter model, optimised model.
Shows that even after optimisation the model keeps a winter peak (NPP-driven):
the seasonal phase cannot be fixed by parameter tuning.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from seapopym.configuration.no_transport import (
    ForcingParameter,
    ForcingUnit,
    FunctionalGroupParameter,
    FunctionalGroupUnit,
    FunctionalTypeParameter,
    KernelParameter,
    MigratoryTypeParameter,
    NoTransportConfiguration,
)
from seapopym.model.no_transport_model import NoTransportModel


def _project_root(marker: str = "pyproject.toml") -> Path:
    for p in [Path.cwd(), *Path.cwd().parents]:
        if (p / marker).exists():
            return p
    raise FileNotFoundError(marker)


ROOT = _project_root()
DATA = ROOT / "data"
FIG = ROOT / "experiment" / "figures"

# --- optimised parameters from the real-data logbook ---
lb = pd.read_parquet(DATA / "ga_logbook_HOT_validation_hotreal.parquet")
best = lb.loc[lb[("Weighted_fitness", "Weighted_fitness")].idxmax(), "Parametre"].to_dict()
print("optimised params:", {k: round(float(v), 4) for k, v in best.items()})

# --- forcing (same construction as the pseudo-obs notebook) ---
raw = xr.open_zarr(DATA / "stations.zarr").load()
ordered = raw.isel(station=raw.station_lat.argsort().values)
y_values = ordered.station_lat.values.astype(np.float32)
x_values = np.array([0.0], dtype=np.float32)


def _to_grid(da: xr.DataArray) -> xr.DataArray:
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
npp.attrs["units"] = "mg/m^2/day"

fg = FunctionalGroupParameter(functional_group=[FunctionalGroupUnit(
    name="zooplankton",
    energy_transfert=best["energy_transfert"],
    functional_type=FunctionalTypeParameter(
        lambda_temperature_0=best["lambda_temperature_0"],
        gamma_lambda_temperature=best["gamma_lambda_temperature"],
        tr_0=best["tr_0"],
        gamma_tr=best["gamma_tr"],
    ),
    migratory_type=MigratoryTypeParameter(day_layer=0, night_layer=0),
)])
config = NoTransportConfiguration(
    forcing=ForcingParameter(temperature=ForcingUnit(forcing=temperature), primary_production=ForcingUnit(forcing=npp)),
    functional_group=fg,
    kernel=KernelParameter(compute_initial_conditions=True),
)
with NoTransportModel.from_configuration(configuration=config) as model:
    model.run()
    model.state.compute()
    biomass = model.state.biomass.load()

opt = biomass.sel(Y=23.0, method="nearest").isel(X=0, functional_group=0).to_pandas().loc["2000-01-01":"2019-12-31"]

# --- obs + reference ---
obs = xr.open_zarr(DATA / "hot_real_observations.zarr")["observed_biomass"].isel(Y=0, X=0).to_pandas().dropna()
ref = xr.open_zarr(DATA / "pseudo_observations.zarr")["observed_biomass"].sel(Y=23.0, method="nearest").isel(X=0).to_pandas()

months = np.arange(1, 13)
obs_m = obs.groupby(obs.index.month).mean().reindex(months)
obs_s = obs.groupby(obs.index.month).std().reindex(months)
ref_m = ref.groupby(ref.index.month).mean().reindex(months)
opt_m = opt.groupby(opt.index.month).mean().reindex(months)

fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
ax.fill_between(months, obs_m - obs_s, obs_m + obs_s, color="#2ca02c", alpha=0.15)
ax.plot(months, obs_m, "o-", color="#2ca02c", lw=2, label="In-situ observations (HOT)")
ax.plot(months, ref_m, "s--", color="#d62728", lw=2, label="SeapoPym (reference params)")
ax.plot(months, opt_m, "^:", color="#9467bd", lw=2, label="SeapoPym (optimised on HOT, NRMSE=1.04)")
ax.set_xticks(months)
ax.set_xticklabels(list("JFMAMJJASOND"))
ax.set_xlabel("Month")
ax.set_ylabel(r"Zooplankton biomass (g C m$^{-2}$)")
ax.set_title("HOT monthly climatology — optimisation cannot fix the seasonal phase", fontsize=10, fontweight="bold")
ax.grid(True, alpha=0.3)
ax.legend(frameon=False)
plt.tight_layout()
fig.savefig(FIG / "HOT_real_climatology_optimized.png", dpi=200, bbox_inches="tight", facecolor="white")
print("wrote", FIG / "HOT_real_climatology_optimized.png")
print("optimised monthly:", {int(m): round(float(opt_m[m]), 3) for m in months})
