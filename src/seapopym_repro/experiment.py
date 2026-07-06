"""Twin-experiment core: the seapopym model wiring + the experiment registry, shared by the
experiment and figure scripts. GA-specific machinery (early stopping, Sobol-init population, the
GA driver) is gone — the GA was abandoned in favour of pycma CMA-ES; only the optimiser-agnostic
forcing / observation / functional-group builders live here.
"""
from __future__ import annotations

import numpy as np
import xarray as xr
from seapopym.configuration.no_transport import ForcingParameter, ForcingUnit
from seapopym_optimization.functional_group import NoTransportFunctionalGroup, Parameter
from seapopym_optimization.functional_group.parameter_initialization import random_uniform_exclusive
from seapopym_optimization.observations import DayCycle, TimeSeriesObservation

from . import paths

# back-compat aliases so scripts can do `from seapopym_repro import experiment as exp; exp.ROOT`
ROOT = paths.ROOT
DATA_DIR = paths.DATA

PARAM_KEYS = ["energy_transfert", "tr_0", "gamma_tr", "lambda_temperature_0", "gamma_lambda_temperature"]
EXPERIMENTS = ["BARENTS", "PAPA", "Bay_of_Biscay", "BATS", "Canaries", "HOT", "MERGED"]
WF = ("Weighted_fitness", "Weighted_fitness")


def build_forcing():
    raw = xr.open_zarr(DATA_DIR / "stations.zarr").load()
    ordered = raw.isel(station=raw.station_lat.argsort().values)
    y = ordered.station_lat.values.astype(np.float32)
    x = np.array([0.0], dtype=np.float32)

    def grid(da):
        arr = da.values.T[:, :, np.newaxis]  # .T transpose, NOT ["T"]
        return xr.DataArray(arr, dims=("T", "Y", "X"),
                            coords={"T": ordered.time.values, "Y": ("Y", y), "X": ("X", x)})

    temperature = grid(ordered.temperature).expand_dims(Z=[0], axis=1)
    npp = grid(ordered.npp)
    for c, a in {"T": "T", "Z": "Z", "Y": "Y", "X": "X"}.items():
        if c in temperature.coords:
            temperature[c].attrs["axis"] = a
        if c in npp.coords:
            npp[c].attrs["axis"] = a
    temperature.attrs["units"] = "degC"
    npp.attrs["units"] = "mg/m^2/day"
    # Cold start: every candidate runs its own 1998-1999 spin-up (the forcing spans 1998-2019) and
    # the cost is scored only on 2000-2019 (the observation window). This matches how the twin target
    # was generated (a cold-start reference run in gen_pseudo_observations, no frozen initial state),
    # so the reference parameters reproduce the target exactly at every station, BARENTS included.
    # The former shared initial_conditions.zarr restart is dropped: it left a decaying restart
    # transient at the slowest station (BARENTS, ~4e-4 on the cost floor).
    return ForcingParameter(
        temperature=ForcingUnit(forcing=temperature),
        primary_production=ForcingUnit(forcing=npp),
    )


def build_observations(experiment, stations_meta):
    pseudo = xr.open_zarr(DATA_DIR / "pseudo_observations.zarr")["observed_biomass"]

    def one(sid):
        obs = pseudo.sel(Y=stations_meta[sid]["lat"], X=0).assign_coords(Z=0).load()
        for c, a in {"T": "T", "Z": "Z", "Y": "Y", "X": "X"}.items():
            if c in obs.coords:
                obs[c].attrs["axis"] = a
        return TimeSeriesObservation(name=sid, observation=obs, observation_type=DayCycle.DAY)

    sids = list(stations_meta) if experiment == "MERGED" else [experiment]
    return [one(sid) for sid in sids]


def functional_groups(bounds):
    return [NoTransportFunctionalGroup(
        name="zooplankton", day_layer=0, night_layer=0,
        energy_transfert=Parameter("energy_transfert", *bounds["energy_transfert"], init_method=random_uniform_exclusive),
        gamma_tr=Parameter("gamma_tr", *bounds["gamma_tr"], init_method=random_uniform_exclusive),
        tr_0=Parameter("tr_0", *bounds["tr_0"], init_method=random_uniform_exclusive),
        gamma_lambda_temperature=Parameter("gamma_lambda_temperature", *bounds["gamma_lambda_temperature"], init_method=random_uniform_exclusive),
        lambda_temperature_0=Parameter("lambda_temperature_0", *bounds["lambda_temperature_0"], init_method=random_uniform_exclusive),
    )]
