"""run_optimization.py — Run a single genetic-algorithm twin experiment.

Per-station or joint (MERGED) optimisation of the SeapoPym biological parameters
against the pseudo-observations produced by
`notebooks/04_twin_experiments/01_generate_pseudo_observations.ipynb`
(Manuscript Section 2.4.4).

Inputs:
    data/stations.zarr             — forcings (T, NPP) for the 6 stations
    data/pseudo_observations.zarr  — reference biomass to fit
    data/initial_conditions.zarr   — spin-up state (biomass, preproduction)
    parameters.yaml                — Table 1 bounds + GA mode (test / production)

Output:
    data/ga_logbook_{STATION}_{MODE}.parquet  — Sobol init + GA generations

Usage:
    python scripts/run_optimization.py --station HOT --mode test
    python scripts/run_optimization.py --station MERGED --mode production
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import xarray as xr
import yaml
from dask.distributed import Client

from seapopym.configuration.no_transport import ForcingParameter, ForcingUnit, KernelParameter
from seapopym.model.no_transport_model import NoTransportSpaceOptimizedLightModel
from seapopym_optimization.algorithm.genetic_algorithm import GeneticAlgorithmParameters
from seapopym_optimization.algorithm.genetic_algorithm.factory import GeneticAlgorithmFactory
from seapopym_optimization.algorithm.genetic_algorithm.logbook import Logbook
from seapopym_optimization.configuration_generator import NoTransportConfigurationGenerator
from seapopym_optimization.cost_function import (
    CostFunction,
    TimeSeriesScoreProcessor,
    nrmse_std_comparator,
    rmse_comparator,  # noqa: F401  (kept for reference; NRMSE is the published metric)
)
from seapopym_optimization.functional_group import (
    FunctionalGroupSet,
    NoTransportFunctionalGroup,
    Parameter,
)
from seapopym_optimization.functional_group.parameter_initialization import random_uniform_exclusive
from seapopym_optimization.observations import DayCycle, TimeSeriesObservation

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

log = logging.getLogger("ga")


def _build_grid(stations_zarr: xr.Dataset) -> tuple[xr.DataArray, xr.DataArray, np.ndarray]:
    """Build (T, Z=1, Y=6, X=1) temperature and (T, Y=6, X=1) NPP forcings.

    Returns (temperature, npp, sorted_y_values).
    """
    order = stations_zarr.station_lat.argsort().values
    ordered = stations_zarr.isel(station=order)
    y_values = ordered.station_lat.values.astype(np.float32)
    x_values = np.array([0.0], dtype=np.float32)

    def to_grid(da: xr.DataArray) -> xr.DataArray:
        arr = da.values.T[:, :, np.newaxis]
        return xr.DataArray(
            arr,
            dims=("T", "Y", "X"),
            coords={"T": ordered.time.values, "Y": ("Y", y_values), "X": ("X", x_values)},
        )

    temperature = to_grid(ordered.temperature).expand_dims(Z=[0], axis=1)
    npp = to_grid(ordered.npp)
    for coord, axis in {"T": "T", "Z": "Z", "Y": "Y", "X": "X"}.items():
        if coord in temperature.coords:
            temperature[coord].attrs["axis"] = axis
        if coord in npp.coords:
            npp[coord].attrs["axis"] = axis
    temperature.attrs["units"] = "degC"
    npp.attrs["units"] = "mg/m^2/day"
    return temperature, npp, y_values


def _build_observations(station: str, stations_meta: dict, obs_file: str = "pseudo_observations.zarr") -> list[TimeSeriesObservation]:
    """Slice the observation dataset into a TimeSeriesObservation list.

    `obs_file` selects which observation zarr to fit against (default: the
    synthetic twin-experiment pseudo-observations). Pass e.g.
    `hot_real_observations.zarr` to optimise against real in-situ data.
    """
    pseudo = xr.open_zarr(DATA_DIR / obs_file)["observed_biomass"]

    def _slice(sid: str) -> TimeSeriesObservation:
        lat = stations_meta[sid]["lat"]
        obs = pseudo.sel(Y=lat, X=0).assign_coords(Z=0).load()
        for coord, axis in {"T": "T", "Z": "Z", "Y": "Y", "X": "X"}.items():
            if coord in obs.coords:
                obs[coord].attrs["axis"] = axis
        return TimeSeriesObservation(name=sid, observation=obs, observation_type=DayCycle.DAY)

    if station == "MERGED":
        return [_slice(sid) for sid in stations_meta]
    return [_slice(station)]


def _build_functional_groups(bounds: dict) -> list[NoTransportFunctionalGroup]:
    """Initialise the zooplankton functional group from Table 1 bounds."""
    return [NoTransportFunctionalGroup(
        name="zooplankton",
        day_layer=0,
        night_layer=0,
        energy_transfert=Parameter(
            "energy_transfert", *bounds["energy_transfert"], init_method=random_uniform_exclusive,
        ),
        gamma_tr=Parameter(
            "gamma_tr", *bounds["gamma_tr"], init_method=random_uniform_exclusive,
        ),
        tr_0=Parameter(
            "tr_0", *bounds["tr_0"], init_method=random_uniform_exclusive,
        ),
        gamma_lambda_temperature=Parameter(
            "gamma_lambda_temperature", *bounds["gamma_lambda_temperature"], init_method=random_uniform_exclusive,
        ),
        lambda_temperature_0=Parameter(
            "lambda_temperature_0", *bounds["lambda_temperature_0"], init_method=random_uniform_exclusive,
        ),
    )]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    parser.add_argument("--station", required=True, help="Station name (BARENTS, PAPA, Bay_of_Biscay, BATS, Canaries, HOT, MERGED)")
    parser.add_argument("--mode", choices=["test", "production", "validation"], default=None,
                        help="Override parameters.yaml mode (default: use whatever is in parameters.yaml).")
    parser.add_argument("--workers", type=int, default=2, help="Number of Dask workers (default: 2).")
    parser.add_argument("--memory-limit", default="8GB", help="Dask memory limit per worker (default: 8GB).")
    parser.add_argument("--obs-file", default="pseudo_observations.zarr",
                        help="Observation zarr in data/ to fit against (default: synthetic pseudo-observations; "
                             "e.g. hot_real_observations.zarr for real in-situ data).")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    with open(ROOT / "parameters.yaml") as f:
        PARAMS = yaml.safe_load(f)
    with open(DATA_DIR / "stations_coords.json") as f:
        STATIONS = json.load(f)

    GA = PARAMS["genetic_algorithm"]
    BOUNDS = PARAMS["model_parameters"]["bounds"]
    mode = args.mode or GA["mode"]
    mode_params = GA[mode]

    # Tag the logbook with the observation source when it is not the default
    # synthetic pseudo-observations, so real-data runs don't overwrite twin runs.
    obs_tag = "" if args.obs_file == "pseudo_observations.zarr" else "_" + Path(args.obs_file).stem.replace("_observations", "").replace("_", "")
    output_path = DATA_DIR / f"ga_logbook_{args.station}_{mode}{obs_tag}.parquet"
    if output_path.exists():
        log.info("Removing stale logbook: %s", output_path)
        output_path.unlink()

    # Forcings (T, NPP at the 6 stations)
    stations_zarr = xr.open_zarr(DATA_DIR / "stations.zarr").load()
    temperature, npp, _ = _build_grid(stations_zarr)

    # Initial conditions (post spin-up state)
    ic = xr.open_zarr(DATA_DIR / "initial_conditions.zarr").load()

    forcing = ForcingParameter(
        temperature=ForcingUnit(forcing=temperature),
        primary_production=ForcingUnit(forcing=npp),
        initial_condition_biomass=ForcingUnit(forcing=ic.biomass),
        initial_condition_production=ForcingUnit(forcing=ic.preproduction),
    )

    observations = _build_observations(args.station, STATIONS, args.obs_file)
    weights = tuple(-1.0 for _ in observations)
    fg_set = FunctionalGroupSet(_build_functional_groups(BOUNDS))

    logbook = Logbook.from_sobol_samples(
        functional_group_parameters=fg_set,
        sample_number=mode_params["population_size"],
        fitness_names=[obs.name for obs in observations],
        saltelli=False,  # plain Sobol sequence of exactly population_size points (= initial population)
    )

    client = Client(n_workers=args.workers, threads_per_worker=1, memory_limit=args.memory_limit)
    log.info("Dask dashboard: %s", client.dashboard_link)
    try:
        cost_function = CostFunction(
            configuration_generator=NoTransportConfigurationGenerator(model_class=NoTransportSpaceOptimizedLightModel),
            functional_groups=fg_set,
            forcing=forcing,
            kernel=KernelParameter(),
            observations=observations,
            processor=TimeSeriesScoreProcessor(comparator=nrmse_std_comparator),
        )

        ga_params = GeneticAlgorithmParameters(
            MUTPB=1.0,
            INDPB=1 / 5,
            ETA=20.0,
            CXPB=GA["cxpb"],
            NGEN=mode_params["n_generations"],
            POP_SIZE=mode_params["population_size"],
            cost_function_weight=weights,
            TOURNSIZE=GA["tournament_size"],
            CX_METHOD=GA["cx_method"],
            CX_ETA=GA["cx_eta"],
            PATIENCE=GA["patience"],
            TOLERANCE=GA["tolerance"],
            MIN_GEN=GA["min_gen"],
        )

        ga = GeneticAlgorithmFactory.create_distributed(
            meta_parameter=ga_params,
            cost_function=cost_function,
            client=client,
            logbook=logbook,
            save=output_path,
        )

        log.info("Station=%s mode=%s | crossover=%s POP=%d NGEN(cap)=%d patience=%s",
                 args.station, mode, GA["cx_method"],
                 mode_params["population_size"], mode_params["n_generations"], GA["patience"])
        ga.optimize()
    finally:
        client.close()

    log.info("Done. Logbook: %s (%.1f KB)", output_path, output_path.stat().st_size / 1024)


if __name__ == "__main__":
    main()
