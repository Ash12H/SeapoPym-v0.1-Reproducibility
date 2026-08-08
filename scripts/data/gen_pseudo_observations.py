"""Generate the synthetic observations the twin experiments fit.

Runs SeapoPym with the reference parameters over the full 1998-2019 forcing, starting from zero
biomass, and keeps the 2000-2019 slice as the target. The run uses the implicit biomass solver, the
same one the optimizer uses, so that an exact recovery stays possible. A target produced with the
other solver would leave a residual cost floor that no parameter set could cross.

Each candidate later runs its own spin-up over the same window, so no initial state is exported.

The result is committed as data/pseudo_observations.zarr. Rerun this script only after changing the
reference parameters in parameters.yaml, and rerun the twin experiments afterwards.

Inputs : data/stations.zarr, parameters.yaml
Output : data/pseudo_observations.zarr   (reference biomass at the six stations)
Run    : .venv/bin/python scripts/data/gen_pseudo_observations.py
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
