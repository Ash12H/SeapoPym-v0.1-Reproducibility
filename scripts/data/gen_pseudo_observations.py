"""rehearsal/gen_pseudo_observations.py — twin-experiment pseudo-observations, generated with
the IMPLICIT biomass solver.

Mirrors notebooks/04_twin_experiments/01_generate_pseudo_observations.ipynb but runs SeapoPym
with `biomass_solver="implicit"`. The twin experiment is self-referential: the optimizer fits with
the implicit solver, so its target (pseudo-obs) MUST be produced with the same solver — otherwise
an NRMSE floor (~0.02-0.036) contaminates recovery.

The reference run is a cold-start integration over the full 1998-2019 forcing; its 2000-2019 slice
is the target. Every candidate later runs its own cold-start spin-up (experiment.build_forcing), so
no frozen initial state is exported: the reference parameters reproduce this target at every station.

  data/pseudo_observations.zarr  (reference biomass at the 6 stations, analysis window)

Run: .venv/bin/python rehearsal/gen_pseudo_observations.py
"""

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import xarray as xr
import yaml
from seapopym.configuration.no_transport import (
    ForcingParameter, ForcingUnit, FunctionalGroupParameter, FunctionalGroupUnit,
    FunctionalTypeParameter, KernelParameter, MigratoryTypeParameter, NoTransportConfiguration,
)
from seapopym.model.no_transport_model import NoTransportModel

SOLVER = "implicit"
from seapopym_repro import paths
ROOT = paths.ROOT
DATA = paths.DATA

PARAMS = yaml.safe_load(open(ROOT / "parameters.yaml"))
REF = PARAMS["model_parameters"]["reference"]
GS = PARAMS["global_simulation"]
T_START = GS["start_date"]
T_ANALYSIS_START = (datetime.strptime(GS["spin_up_end"], "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
T_END = "2019-12-31"
print(f"solver={SOLVER} | sim {T_START} -> {T_END} | pseudo-obs window {T_ANALYSIS_START} -> {T_END}")

# --- (T, Z=1, Y=6, X=1) forcing grid from station time series --------------------------------
raw = xr.open_zarr(DATA / "stations.zarr").sel(time=slice(T_START, T_END)).load()
ordered = raw.isel(station=raw.station_lat.argsort().values)
y_values = ordered.station_lat.values.astype(np.float32)
x_values = np.array([0.0], dtype=np.float32)


def _to_grid(da):
    return xr.DataArray(da.values.T[:, :, np.newaxis], dims=("T", "Y", "X"),
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

# --- reference run with the implicit solver + initial-condition export ------------------------
fg = FunctionalGroupParameter(functional_group=[FunctionalGroupUnit(
    name="zooplankton",
    energy_transfert=REF["energy_transfert"],
    functional_type=FunctionalTypeParameter(
        lambda_temperature_0=REF["lambda_temperature_0"],
        gamma_lambda_temperature=REF["gamma_lambda_temperature"],
        tr_0=REF["tr_0"], gamma_tr=REF["gamma_tr"],
    ),
    migratory_type=MigratoryTypeParameter(day_layer=0, night_layer=0),
)])
config = NoTransportConfiguration(
    forcing=ForcingParameter(
        temperature=ForcingUnit(forcing=temperature),
        primary_production=ForcingUnit(forcing=npp),
    ),
    functional_group=fg,
    kernel=KernelParameter(compute_initial_conditions=True, biomass_solver=SOLVER),
)
with NoTransportModel.from_configuration(configuration=config) as model:
    model.run()
    model.state.compute()
    biomass = model.state.biomass.load().copy()

# --- persist --------------------------------------------------------------------------------
pseudo_obs_path = DATA / "pseudo_observations.zarr"

biomass_obs = biomass.sel(T=slice(T_ANALYSIS_START, T_END), functional_group=0, drop=True)
biomass_obs.to_dataset(name="observed_biomass").to_zarr(pseudo_obs_path, mode="w", zarr_format=2)

mb = sum(f.stat().st_size for f in pseudo_obs_path.rglob("*") if f.is_file()) / 1e6
print(f"  wrote {pseudo_obs_path.name}: {mb:.2f} MB")
print("done (implicit pseudo-observations + initial conditions).")
