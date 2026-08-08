"""Run the twin experiments: recover the five parameters from synthetic observations.

Runs a multi-start CMA-ES at each station and on the joint MERGED experiment, and reports both the
cost reached and the parameters recovered. The search is driven by the framework optimizer
seapopym_optimization.algorithm.CMAES, which wraps pycma.

The script defaults are the published configuration: a population of int(4 + 3*ln(5)) = 8, twenty
random restarts per experiment (seeds 0 to 19), the implicit biomass solver, and a cost equal to the
mean station NRMSE normalized by the mean of the target series. The search stops on pycma's standard
criteria, and the iteration cap set by --ngen is only a backstop, which the cap_hit column and the
logged stop reason confirm was never reached.

The parameters recovered, not the cost, are what the experiment measures. A low cost at a warm
station reflects equifinality rather than recovery.

Each restart writes its full trajectory to its own logbook and a rerun reuses the restarts already
computed, so an interruption costs at most the restart in flight.

Inputs : data/pseudo_observations.zarr, data/stations.zarr, parameters.yaml
Outputs: products/cmaes_seed_ensemble_l8_nrmse_mean.csv        one row per experiment and restart
         products/cmaes_convergence_traces_l8_nrmse_mean.csv   best-so-far cost per evaluation
         results_raw/cmaes/                                    per-restart trajectories
Run    : .venv/bin/python scripts/experiments/run_cmaes_seed_ensemble.py
"""

from __future__ import annotations

import argparse
import json
import shutil
import time

import numpy as np
import pandas as pd
import yaml
from dask.distributed import Client
from seapopym.configuration.no_transport import KernelParameter
from seapopym.model.no_transport_model import NoTransportSpaceOptimizedLightModel
from seapopym_optimization.algorithm import CMAES, CMAESParameters
from seapopym_optimization.algorithm.genetic_algorithm.evaluation_strategies import DistributedEvaluation
from seapopym_optimization.configuration_generator import NoTransportConfigurationGenerator
from seapopym_optimization.cost_function import CostFunction, TimeSeriesScoreProcessor, nrmse_std_comparator
from seapopym_optimization.functional_group import FunctionalGroupSet

from seapopym_repro import experiment as ga
from seapopym_repro import paths
from seapopym_repro.metrics import COMPARATORS

# Cost comparators. The published runs use PRODUCTION_METRIC (nrmse_mean, the NRMSE normalized by the
# mean of the target series); the others are available for alternative-cost runs, which are namespaced
# by metric so they never overwrite the published products.
COMP = {**COMPARATORS, "nrmse_std": nrmse_std_comparator}

EXPERIMENTS = ["BARENTS", "PAPA", "Bay_of_Biscay", "BATS", "Canaries", "HOT", "MERGED"]
SIGMA0, NGEN = 0.30, 1000                            # NGEN is a backstop; convergence is sigma<1e-3 (set by main)
PRODUCTS = paths.PRODUCTS                            # committed CSV products (recovery + traces) the figures read
CMA_DIR = paths.RESULTS_RAW / "cmaes"                # heavy intermediates (gitignored): per-seed parquets + best copy
CMA_SEED_DIR = CMA_DIR / "seeds"                     # per-seed trajectories (resume unit), lambda-namespaced
REF = yaml.safe_load(open(ga.ROOT / "parameters.yaml"))["model_parameters"]["reference"]


def cmaes_lambda(n_params: int) -> int:
    """CMA-ES default population size, lambda = int(4 + 3*ln(N))."""
    return int(4 + 3 * np.log(n_params))


def run_seed(seed, cost, evalstrat, lam, ngen):
    """One seeded CMA-ES restart, driven by the framework optimizer.

    The search is delegated to `seapopym_optimization.algorithm.CMAES`, the framework's first-class
    CMA-ES backend, so this experiment exercises the same pluggable optimizer interface as the rest
    of the framework rather than a private pycma loop. The optimizer starts from a reproducible
    random point in [0,1]^5 (`RandomState(seed)`), lets pycma handle the box bounds smoothly (no hard
    clipping -> no boundary pile-up), and stops on pycma's STANDARD criteria (tolx/tolfun/
    conditioncov/...) with maxiter=ngen as a backstop.

    The cost weights are uniform and negative, one per observation, so the minimized scalar is the
    mean NRMSE across the observations of the experiment (a single station, or the six of MERGED).
    Returns (best, params, logbook, stop_reason).
    """
    weights = (-1.0,) * len(cost.observations)          # uniform -> the minimized scalar is the mean NRMSE
    cmaes = CMAES(
        CMAESParameters(NGEN=ngen, SIGMA0=SIGMA0, POP_SIZE=lam, SEED=seed, cost_function_weight=weights),
        cost,
        evalstrat,
    )
    df = cmaes.optimize()
    wf = df[("Weighted_fitness", "Weighted_fitness")]
    bx = df.loc[wf.idxmax(), "Parametre"].to_dict()
    return float(-wf.max()), {k: float(bx[k]) for k in ga.PARAM_KEYS}, df, cmaes.stop_reason


def best_of_logbook(out):
    """Read the best cost, the best parameters and the last generation from an existing logbook."""
    df = pd.read_parquet(out)
    wf = df[("Weighted_fitness", "Weighted_fitness")]
    b = df.loc[wf.idxmax(), "Parametre"].to_dict()
    return float(-wf.max()), {k: float(b[k]) for k in ga.PARAM_KEYS}, int(df.index.get_level_values("Generation").max())


def build_convergence_traces(lam, experiments, mtag="", ndown=120):
    """Build the convergence-traces product from the per-restart logbooks.

    For each restart, takes the best-so-far cost per generation, converts generations to model
    evaluations as (generation + 1) * lam, and keeps at most `ndown` log-spaced points. The figure
    has a logarithmic x axis, so the subsampling is not visible, and the result is small enough to
    commit while the full trajectories stay out of the repository.

    Columns: experiment, seed, evaluations, best_nrmse, the last one holding the cost whatever the
    metric. Written to products/cmaes_convergence_traces_l{lam}{mtag}.csv.
    """
    wf = ("Weighted_fitness", "Weighted_fitness")
    rows = []
    for exp in experiments:
        for p in sorted(CMA_SEED_DIR.glob(f"ga_logbook_{exp}_lambda{lam}{mtag}_seed*.parquet")):
            seed = int(p.stem.split("seed")[-1])
            per_gen = pd.read_parquet(p)[wf].groupby(level="Generation").max().sort_index()
            best = -per_gen.cummax().to_numpy()
            evals = (per_gen.index.to_numpy() + 1) * lam
            n = len(best)
            if n > ndown:                       # log-spaced indices: dense early, sparse late
                idx = np.unique(np.round(np.logspace(0, np.log10(n), ndown)).astype(int) - 1)
                idx = idx[(idx >= 0) & (idx < n)]
            else:
                idx = np.arange(n)
            for i in idx:
                rows.append({"experiment": exp, "seed": seed,
                             "evaluations": int(evals[i]), "best_nrmse": float(best[i])})
    out = PRODUCTS / f"cmaes_convergence_traces_l{lam}{mtag}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"froze convergence traces -> {out.name} ({len(rows)} rows, "
          f"{len({(r['experiment'], r['seed']) for r in rows})} traces)", flush=True)
    return out


def run_experiment(exp, params, stations_meta, forcing, client, lam, seeds, ngen, comparator, mtag):
    observations = ga.build_observations(exp, stations_meta)
    fg_set = FunctionalGroupSet(ga.functional_groups(params["model_parameters"]["bounds"]))
    cost = CostFunction(
        configuration_generator=NoTransportConfigurationGenerator(model_class=NoTransportSpaceOptimizedLightModel),
        functional_groups=fg_set, forcing=forcing, kernel=KernelParameter(biomass_solver="implicit"),
        observations=observations, processor=TimeSeriesScoreProcessor(comparator=comparator),
    )
    # scatter big read-only data to workers ONCE (broadcast); avoids per-generation re-transmit ->
    # memory growth + leaked semaphores. Mirrors GeneticAlgorithmFactory.create_distributed.
    cost.forcing = client.scatter(cost.forcing, broadcast=True)
    for _name in list(cost.observations):
        cost.observations[_name] = client.scatter(cost.observations[_name], broadcast=True)
    evalstrat = DistributedEvaluation(cost, client)

    rows, t0 = [], time.time()
    for seed in seeds:
        out = CMA_SEED_DIR / f"ga_logbook_{exp}_lambda{lam}{mtag}_seed{seed}.parquet"  # namespaced by lambda + metric
        if out.exists():                              # per-seed RESUME (robust restart signal)
            nrmse, prm, stop_gen = best_of_logbook(out)
            stop_reason = "reuse"
        else:
            nrmse, prm, df, stop_reason = run_seed(seed, cost, evalstrat, lam, ngen)
            df.to_parquet(out)
            stop_gen = int(df.index.get_level_values("Generation").max())
        cap = stop_gen >= ngen - 1
        rows.append({"experiment": exp, "seed": seed, "lambda": lam, "best_nrmse": nrmse,
                     "stop_gen": stop_gen, "cap_hit": cap, **prm})
        print(f"    {exp:14s} seed {seed:3d} (λ={lam}) | NRMSE={nrmse:.4f} | stop_gen={stop_gen}"
              + (" CAP!" if cap else "") + f" | stop={stop_reason}", flush=True)
    # copy the best seed's logbook to the canonical name read by the figure scripts (metric-namespaced)
    best = min(rows, key=lambda r: r["best_nrmse"])
    shutil.copyfile(CMA_SEED_DIR / f"ga_logbook_{exp}_lambda{lam}{mtag}_seed{best['seed']}.parquet",
                    CMA_DIR / f"ga_logbook_{exp}{mtag}.parquet")
    vals = [r["best_nrmse"] for r in rows]
    print(f"  {exp:14s} | best={min(vals):.4f} median={np.median(vals):.4f} max={max(vals):.4f} "
          f"| cap_hit={sum(r['cap_hit'] for r in rows)}/{len(rows)} | {(time.time()-t0)/60:.1f} min", flush=True)
    return rows


def main():
    global NGEN
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiments", default=",".join(EXPERIMENTS))
    ap.add_argument("--lambda", dest="lam", type=int, default=8,
                    help="CMA-ES population (default 8 = DEAP default int(4+3*ln(N)) for 5 params)")
    ap.add_argument("--n-seeds", type=int, default=20, help="run seeds 0..n-1 (default 20, the published ensemble)")
    ap.add_argument("--seeds", default=None, help="explicit comma-separated seeds (overrides --n-seeds)")
    ap.add_argument("--ngen", type=int, default=1000,
                    help="iteration-cap backstop (default 1000); convergence is sigma<1e-3, so the cap should NOT bind")
    ap.add_argument("--workers", type=int, default=8, help="process workers, single station (default 8)")
    ap.add_argument("--memory-limit", default="4GB", help="per-worker memory, single station (default 4GB)")
    ap.add_argument("--merged-workers", type=int, default=6, help="process workers for MERGED (default 6)")
    ap.add_argument("--merged-mem", default="6GB", help="per-worker memory for MERGED (default 6GB)")
    ap.add_argument("--metric", default=paths.PRODUCTION_METRIC, choices=sorted(COMP),
                    help=f"cost comparator (default {paths.PRODUCTION_METRIC} = the published products); "
                         "any other value namespaces the outputs by metric")
    ap.add_argument("--freeze", action="store_true",
                    help="rebuild the committed products (convergence-traces CSV) from existing per-seed logbooks; no optimisation")
    args = ap.parse_args()
    experiments = [e.strip() for e in args.experiments.split(",") if e.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()] if args.seeds else list(range(args.n_seeds))
    lam = args.lam
    NGEN = args.ngen                                          # used by run_seed (module global)
    metric = args.metric                                     # cost comparator key
    comparator = COMP[metric]
    mtag = "" if metric == "nrmse_std" else f"_{metric}"      # default keeps the existing filenames byte-stable
    results_csv = PRODUCTS / f"cmaes_seed_ensemble_l{lam}{mtag}.csv"   # committed product, namespaced by lambda + metric

    PRODUCTS.mkdir(parents=True, exist_ok=True)
    CMA_DIR.mkdir(parents=True, exist_ok=True)
    CMA_SEED_DIR.mkdir(parents=True, exist_ok=True)
    if args.freeze:                                          # rebuild committed products only (no runs)
        build_convergence_traces(lam, experiments, mtag)
        return
    params = yaml.safe_load(open(ga.ROOT / "parameters.yaml"))
    stations_meta = json.load(open(ga.DATA_DIR / "stations_coords.json"))
    forcing = ga.build_forcing()
    print(f"CMA-ES ensemble: λ={lam} | metric={metric} | NGEN cap={NGEN} | {len(seeds)} seeds | experiments={experiments}", flush=True)
    print(f"  per-seed resume on results_raw/cmaes/seeds/ (lambda+metric-namespaced) | CSV -> {results_csv.name}", flush=True)

    all_rows = []
    for exp in experiments:  # fresh client PER EXPERIMENT (releases workers + scattered data between exps)
        client = (Client(n_workers=args.merged_workers, threads_per_worker=1, memory_limit=args.merged_mem)
                  if exp == "MERGED"
                  else Client(n_workers=args.workers, threads_per_worker=1, memory_limit=args.memory_limit))
        try:
            all_rows.extend(run_experiment(exp, params, stations_meta, forcing, client, lam, seeds, NGEN, comparator, mtag))
            pd.DataFrame(all_rows).to_csv(results_csv, index=False)   # checkpoint after each experiment
        finally:
            client.close()

    df = pd.DataFrame(all_rows)
    df.to_csv(results_csv, index=False)
    g = df.groupby("experiment").agg(best=("best_nrmse", "min"), median=("best_nrmse", "median"),
                                     worst=("best_nrmse", "max"), cap_hits=("cap_hit", "sum"), n=("seed", "count"))
    print(f"\n===== CMA-ES ensemble (λ={lam}, {len(seeds)} seeds) =====", flush=True)
    print(g.to_string(float_format=lambda x: f"{x:.4f}"), flush=True)
    total_cap = int(df["cap_hit"].sum())
    print(f"\nseeds hitting NGEN cap ({NGEN}): {total_cap}/{len(df)} "
          + ("(none, convergence governs)" if total_cap == 0 else "(some capped → raise --ngen)"), flush=True)
    build_convergence_traces(lam, experiments, mtag)         # freeze the committed convergence product


if __name__ == "__main__":
    main()
