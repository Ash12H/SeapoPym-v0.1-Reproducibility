"""rehearsal/run_seed_ensemble.py — seeded GA restart ensemble (reliability) for ALL experiments.

The single-run GA result is seed-dependent (e.g. BARENTS NRMSE 0.035 vs 0.072 across seeds): on
low-information regimes the GA converges prematurely to different basins. Reporting ONE run is a
lucky/unlucky draw. Instead this runs the frozen GA config (run_ga_production.CFG, pop 64) across
N seeds per experiment and reports the DISTRIBUTION of the best NRMSE + the recovered-parameter
cloud. The run-to-run spread is itself the result: wide on warm/oligotrophic stations
(non-identifiable), tight on cold BARENTS, tighter on MERGED (the joint remedy).

Each seed fully seeds the Sobol init scramble AND the DEAP operators (reproducible restarts).

Dask: one THREADED in-process client per experiment (processes=False) + a one-shot broadcast
scatter of forcing/observations reused across that experiment's seeds. Recreating the client per
experiment releases all scattered data between experiments -> no cross-experiment memory growth,
no semaphore leak (the failure mode of the old process-based runs).

Outputs (none overwrite the canonical ga_logbook_{exp}.parquet read by the figures):
    logbooks/seeds/ga_logbook_{exp}_seed{S}.parquet   per-seed logbook (source of truth)
    logbooks/seed_ensemble.csv                        one row per (exp, seed): best_nrmse + 5 params
Resumable: (exp, seed) pairs already in the CSV are skipped; an existing per-seed logbook is reused.

Run:
    .venv/bin/python rehearsal/run_seed_ensemble.py                       # all 7, seeds 0-9
    .venv/bin/python rehearsal/run_seed_ensemble.py --experiments BARENTS,MERGED --seeds 0,1
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from dask.distributed import Client

import run_ga_production as ga
from seapopym.configuration.no_transport import KernelParameter
from seapopym.model.no_transport_model import NoTransportSpaceOptimizedLightModel
from seapopym_optimization.algorithm.genetic_algorithm import GeneticAlgorithmParameters
from seapopym_optimization.algorithm.genetic_algorithm.factory import GeneticAlgorithmFactory
from seapopym_optimization.configuration_generator import NoTransportConfigurationGenerator
from seapopym_optimization.cost_function import CostFunction, TimeSeriesScoreProcessor, nrmse_std_comparator
from seapopym_optimization.functional_group import FunctionalGroupSet

ROOT, OUT = ga.ROOT, ga.OUT_DIR
SEED_DIR = OUT / "seeds"
RESULTS = OUT / "seed_ensemble.csv"
REF = yaml.safe_load(open(ROOT / "parameters.yaml"))["model_parameters"]["reference"]


def best_of(out: Path) -> tuple[float, dict]:
    d = pd.read_parquet(out)
    b = d.loc[d[ga.WF].idxmax(), "Parametre"].to_dict()
    return float(-d[ga.WF].max()), {k: float(b[k]) for k in ga.PARAM_KEYS}


def run_seed(exp, seed, fg_set, observations, cost, client) -> dict:
    """One seeded GA restart on a cost whose forcing/observations are ALREADY scattered (Futures)."""
    out = SEED_DIR / f"ga_logbook_{exp}_seed{seed}.parquet"
    if out.exists():  # reuse an already-computed per-seed logbook
        nrmse, best = best_of(out)
        print(f"  {exp:14s} seed {seed} | reuse existing logbook | NRMSE={nrmse:.5g}", flush=True)
        return {"experiment": exp, "seed": seed, "best_nrmse": nrmse, "stop_gen": -1, "minutes": 0.0, **best}
    random.seed(seed)
    np.random.seed(seed)
    logbook = ga.sobol_init_logbook(fg_set, ga.CFG["pop"], [o.name for o in observations], seed=seed)
    ga_params = GeneticAlgorithmParameters(
        MUTPB=1.0, INDPB=1 / 5, ETA=20.0, CXPB=0.9,
        NGEN=ga.CFG["ngen_cap"], POP_SIZE=ga.CFG["pop"], cost_function_weight=tuple(-1.0 for _ in observations),
        TOURNSIZE=2, CX_METHOD=ga.CFG["crossover"], CX_ETA=ga.CFG["cx_eta"],
    )
    # create_distributed detects the already-scattered Futures on `cost` and reuses them (no re-scatter).
    g = GeneticAlgorithmFactory.create_distributed(
        meta_parameter=ga_params, cost_function=cost, client=client, logbook=logbook, save=out)
    print(f"  {exp:14s} seed {seed} | SBX pop={ga.CFG['pop']} cap={ga.CFG['ngen_cap']}", flush=True)
    t0 = time.time()
    _, stop_gen, best_cost = ga.optimize_with_early_stopping(g, ga.CFG["patience"], ga.CFG["tol"], ga.CFG["min_gen"])
    dt = (time.time() - t0) / 60
    _, best = best_of(out)
    print(f"     {exp:14s} seed {seed} done {dt:.1f} min | stop_gen={stop_gen} | NRMSE={best_cost:.5g}", flush=True)
    return {"experiment": exp, "seed": seed, "best_nrmse": best_cost, "stop_gen": stop_gen, "minutes": dt, **best}


def make_client(exp, args) -> Client:
    """Fresh Dask client PER EXPERIMENT (releases workers + scattered data between experiments,
    which is what prevented the old single-long-lived-client run from leaking itself dead ~1 exp in).

    Default = process workers (true parallelism: each 0D eval is GIL-bound, so threads serialise and
    are ~5x slower). MERGED is memory-heavy -> fewer workers x more memory. --threaded forces a single
    in-process scheduler (no semaphore/OOM risk, but ~5x slower) as an escape hatch.
    """
    if args.threaded:
        return Client(processes=False, n_workers=1, threads_per_worker=args.threads)
    if exp == "MERGED":
        return Client(n_workers=args.merged_workers, threads_per_worker=1, memory_limit=args.merged_mem)
    return Client(n_workers=args.workers, threads_per_worker=1, memory_limit=args.memory_limit)


def run_experiment(exp, seeds, params, stations_meta, forcing, args) -> list[dict]:
    """One fresh client for this experiment; scatter forcing/obs once, reuse across seeds."""
    observations = ga.build_observations(exp, stations_meta)
    fg_set = FunctionalGroupSet(ga.functional_groups(params["model_parameters"]["bounds"]))
    cost = CostFunction(
        configuration_generator=NoTransportConfigurationGenerator(model_class=NoTransportSpaceOptimizedLightModel),
        functional_groups=fg_set, forcing=forcing, kernel=KernelParameter(biomass_solver="implicit"),
        observations=observations, processor=TimeSeriesScoreProcessor(comparator=nrmse_std_comparator),
    )
    client = make_client(exp, args)
    rows = []
    try:
        # scatter ONCE (broadcast) so every seed/generation reuses the same Futures
        cost.forcing = client.scatter(cost.forcing, broadcast=True)
        for name in list(cost.observations):
            cost.observations[name] = client.scatter(cost.observations[name], broadcast=True)
        for seed in seeds:
            row = run_seed(exp, seed, fg_set, observations, cost, client)
            rows.append(row)
            pd.DataFrame([row]).to_csv(RESULTS, mode="a", header=not RESULTS.exists(), index=False)
    finally:
        client.close()
    return rows


def report(df: pd.DataFrame) -> None:
    print("\n===== seed ensemble (pop {}, {} seeds) — best NRMSE per experiment =====".format(
        ga.CFG["pop"], df.groupby("experiment").seed.nunique().max()), flush=True)
    g = df.groupby("experiment").best_nrmse.agg(["median", "min", "max", "std", "count"])
    g["spread"] = g["max"] - g["min"]
    # keep the manuscript experiment order, restricted to what's present
    order = [e for e in ga.EXPERIMENTS if e in g.index]
    print(g.loc[order].to_string(float_format=lambda x: f"{x:.4f}"), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiments", default=",".join(ga.EXPERIMENTS))
    ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--workers", type=int, default=8, help="process workers, single-station (default 8)")
    ap.add_argument("--memory-limit", default="4GB", help="per-worker memory, single-station (default 4GB)")
    ap.add_argument("--merged-workers", type=int, default=6, help="process workers for MERGED (default 6)")
    ap.add_argument("--merged-mem", default="6GB", help="per-worker memory for MERGED (default 6GB)")
    ap.add_argument("--threaded", action="store_true",
                    help="escape hatch: single in-process threaded scheduler (stable but ~5x slower)")
    ap.add_argument("--threads", type=int, default=8, help="threads when --threaded (default 8)")
    args = ap.parse_args()
    experiments = [e.strip() for e in args.experiments.split(",") if e.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    SEED_DIR.mkdir(parents=True, exist_ok=True)
    params = yaml.safe_load(open(ROOT / "parameters.yaml"))
    stations_meta = json.load(open(ga.DATA_DIR / "stations_coords.json"))
    forcing = ga.build_forcing()

    done = set()
    if RESULTS.exists():
        d = pd.read_csv(RESULTS)
        done = set(zip(d.experiment, d.seed))
        print(f"resuming: {len(done)} (experiment, seed) pairs already done", flush=True)

    print(f"GA seed ensemble | pop={ga.CFG['pop']} | seeds={seeds} | experiments={experiments}", flush=True)
    for exp in experiments:  # MERGED last (heaviest)
        todo = [s for s in seeds if (exp, s) not in done]
        if not todo:
            print(f"  {exp:14s} | all seeds done, skipping", flush=True)
            continue
        run_experiment(exp, todo, params, stations_meta, forcing, args)

    report(pd.read_csv(RESULTS))


if __name__ == "__main__":
    main()
