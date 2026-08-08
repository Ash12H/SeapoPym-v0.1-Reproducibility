"""Run one station and one restart of the twin experiment, as a check.

Reproduces a single (station, restart) pair of run_cmaes_seed_ensemble.py, with the same forcing,
synthetic observations, functional groups and cost function, and prints the recovered parameters
next to their reference values. Useful to check an installation, or to inspect one restart without
running the full ensemble.

Run: .venv/bin/python scripts/experiments/validate_cmaes_framework_station.py [STATION] [SEED]
"""
from __future__ import annotations

import json
import sys

import yaml
from dask.distributed import Client

from seapopym.configuration.no_transport import KernelParameter
from seapopym.model.no_transport_model import NoTransportSpaceOptimizedLightModel
from seapopym_optimization.algorithm import CMAES, CMAESParameters
from seapopym_optimization.algorithm.genetic_algorithm.evaluation_strategies import DistributedEvaluation
from seapopym_optimization.configuration_generator import NoTransportConfigurationGenerator
from seapopym_optimization.cost_function import CostFunction, TimeSeriesScoreProcessor
from seapopym_optimization.functional_group import FunctionalGroupSet

from seapopym_repro import experiment as ga, paths
from seapopym_repro.metrics import COMPARATORS

STATION = sys.argv[1] if len(sys.argv) > 1 else "BARENTS"
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 0
METRIC = paths.PRODUCTION_METRIC
NGEN = 1000


def main() -> None:
    params = yaml.safe_load(open(ga.ROOT / "parameters.yaml"))
    ref = params["model_parameters"]["reference"]
    stations_meta = json.load(open(ga.DATA_DIR / "stations_coords.json"))
    forcing = ga.build_forcing()
    observations = ga.build_observations(STATION, stations_meta)
    fg_set = FunctionalGroupSet(ga.functional_groups(params["model_parameters"]["bounds"]))
    cost = CostFunction(
        configuration_generator=NoTransportConfigurationGenerator(model_class=NoTransportSpaceOptimizedLightModel),
        functional_groups=fg_set,
        forcing=forcing,
        kernel=KernelParameter(biomass_solver="implicit"),
        observations=observations,
        processor=TimeSeriesScoreProcessor(comparator=COMPARATORS[METRIC]),
    )

    client = Client(n_workers=8, threads_per_worker=1, memory_limit="4GB")
    try:
        cost.forcing = client.scatter(cost.forcing, broadcast=True)              # broadcast read-only data once
        for name in list(cost.observations):
            cost.observations[name] = client.scatter(cost.observations[name], broadcast=True)
        evalstrat = DistributedEvaluation(cost, client)
        print(f"station={STATION} seed={SEED} metric={METRIC} n_obs={len(cost.observations)}", flush=True)

        cmaes = CMAES(CMAESParameters(NGEN=NGEN, SEED=SEED), cost, evalstrat)     # the framework optimizer
        logbook = cmaes.optimize()
    finally:
        client.close()

    wf = ("Weighted_fitness", "Weighted_fitness")
    best_cost = float(-logbook[wf].max())
    best = logbook.loc[logbook[wf].idxmax(), "Parametre"].to_dict()
    generations = int(logbook.index.get_level_values("Generation").max()) + 1
    print(f"\nbest {METRIC} = {best_cost:.5f}  |  generations = {generations}  |  logbook rows = {len(logbook)}")
    print("recovered parameter        value      (reference,   rel.err)")
    for key in ga.PARAM_KEYS:
        value, reference = float(best[key]), float(ref[key])
        print(f"  {key:26s} {value:10.4f}  (ref {reference:9.4f}, {(value - reference) / abs(reference) * 100:+6.1f}%)")


if __name__ == "__main__":
    main()
