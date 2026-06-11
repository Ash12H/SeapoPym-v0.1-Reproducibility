"""rehearsal/run_cmaes_seed_ensemble.py — seeded CMA-ES ensemble (pycma) for the twin experiment.

Per-station seeded multi-start CMA-ES (Hansen's reference implementation, pycma): characterises
convergence + identifiability from the distribution of N restarts, and serves as the
operator-independent cross-check of the GA (same objective, different optimiser family).

Config (defaults): lambda = CMA-ES default int(4+3*ln(N)) = 8, 100 fully-random restarts per
experiment (seeds 0..99). pycma handles the [0,1]^5 box bounds smoothly (no hard clipping ->
no boundary pile-up, unlike the old DEAP+clip version which trapped solutions on the bounds).
Termination = pycma's STANDARD criteria (tolx 1e-4 / tolfun / conditioncov / ...); maxiter=ngen
(default 1000) is a backstop that should NOT bind — cap_hit + the logged stop reason verify this.
Same objective as the GA (mean station NRMSE via CostFunction + DistributedEvaluation, Dask method A:
the lambda candidates of each generation evaluated across process workers). Compare on PARAMETER
RECOVERY vs reference, not on NRMSE (low NRMSE on warm stations is equifinality, not recovery).

RESUMABLE per seed: each seed's full trajectory is written to a lambda-namespaced logbook; a restart
reuses any seed whose logbook already exists (a crash loses at most the in-flight seed). Outputs go to
rehearsal/cmaes/ (the dedicated production folder), namespaced by lambda:
    cmaes/seeds/ga_logbook_{exp}_lambda{L}_seed{S}.parquet   per-seed trajectory (resume unit)
    cmaes/ga_logbook_{exp}.parquet                           best seed (read by figure scripts)
    cmaes/cmaes_seed_ensemble_l{L}.csv                       all (exp, seed) results + cap_hit

Run: .venv/bin/python rehearsal/run_cmaes_seed_ensemble.py
"""

from __future__ import annotations

import argparse
import json
import shutil
import time

import numpy as np
import pandas as pd
import yaml
import cma  # pycma — Hansen's reference CMA-ES (standard termination criteria + smooth bound handling)
from dask.distributed import Client

from seapopym_repro import experiment as ga, paths
from seapopym.configuration.no_transport import KernelParameter
from seapopym.model.no_transport_model import NoTransportSpaceOptimizedLightModel
from seapopym_optimization.algorithm.genetic_algorithm.evaluation_strategies import DistributedEvaluation
from seapopym_optimization.configuration_generator import NoTransportConfigurationGenerator
from seapopym_optimization.cost_function import CostFunction, TimeSeriesScoreProcessor, nrmse_std_comparator
from seapopym_optimization.functional_group import FunctionalGroupSet

EXPERIMENTS = ["BARENTS", "PAPA", "Bay_of_Biscay", "BATS", "Canaries", "HOT", "MERGED"]
SIGMA0, NGEN = 0.30, 1000                            # NGEN is a backstop; convergence is sigma<1e-3 (set by main)
PRODUCTS = paths.PRODUCTS                            # committed CSV products (recovery + traces) the figures read
CMA_DIR = paths.RESULTS_RAW / "cmaes"                # heavy intermediates (gitignored): per-seed parquets + best copy
CMA_SEED_DIR = CMA_DIR / "seeds"                     # per-seed trajectories (resume unit), lambda-namespaced
REF = yaml.safe_load(open(ga.ROOT / "parameters.yaml"))["model_parameters"]["reference"]


def cmaes_lambda(n_params: int) -> int:
    """CMA-ES default population size, lambda = int(4 + 3*ln(N))."""
    return int(4 + 3 * np.log(n_params))


def run_seed(seed, names, lo, hi, obs_names, evalstrat, lam, ngen):
    """One seeded CMA-ES restart (pycma), evaluated via `evalstrat` (Dask method A).

    Fresh random start in [0,1]^5 (reproducible from `seed`); pycma handles the box bounds smoothly
    (no hard clipping -> no boundary pile-up). Stops on pycma's STANDARD criteria (tolx/tolfun/
    conditioncov/...) with maxiter=ngen as a backstop. Returns (best, params, logbook, stop_reason).
    """
    D = len(names)

    def denorm(xn):
        return (lo + np.clip(np.asarray(xn, float), 0.0, 1.0) * (hi - lo)).tolist()

    x0 = np.random.RandomState(seed).uniform(0.0, 1.0, D).tolist()   # reproducible random restart
    es = cma.CMAEvolutionStrategy(x0, SIGMA0, {
        "bounds": [0.0, 1.0], "popsize": lam, "seed": int(seed) + 1,  # +1: pycma treats seed 0 as random
        "maxiter": ngen, "tolx": 1e-4, "verbose": -9})
    rows, index, best, gen = [], [], np.inf, 0
    while not es.stop():
        X = es.ask()                                   # lam candidates in [0,1]^D (bounds handled by pycma)
        reals = [denorm(x) for x in X]
        fits = evalstrat.evaluate(reals)
        costs = [float(np.mean(np.asarray(f, float))) for f in fits]
        costs = [c if np.isfinite(c) else 1e6 for c in costs]   # nan/inf -> penalty (safe for es.tell)
        es.tell(X, costs)
        for i, (real, f, c) in enumerate(zip(reals, fits, costs)):
            index.append((gen, False, i))
            row = {("Parametre", p): real[j] for j, p in enumerate(names)}
            row.update({("Fitness", o): float(np.asarray(f, float)[k]) for k, o in enumerate(obs_names)})
            row[("Weighted_fitness", "Weighted_fitness")] = -c
            rows.append(row)
        best = min(best, min(costs))
        gen += 1
    df = pd.DataFrame(rows, index=pd.MultiIndex.from_tuples(
        index, names=["Generation", "Is_From_Previous_Generation", "Individual"]))
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    bx = df.loc[df[("Weighted_fitness", "Weighted_fitness")].idxmax(), "Parametre"].to_dict()
    return best, {k: float(bx[k]) for k in ga.PARAM_KEYS}, df, ",".join(sorted(es.stop()))


def best_of_logbook(out):
    """(best NRMSE, best params, stop_gen) from an existing per-seed logbook — for resume."""
    df = pd.read_parquet(out)
    wf = df[("Weighted_fitness", "Weighted_fitness")]
    b = df.loc[wf.idxmax(), "Parametre"].to_dict()
    return float(-wf.max()), {k: float(b[k]) for k in ga.PARAM_KEYS}, int(df.index.get_level_values("Generation").max())


def build_convergence_traces(lam, experiments, ndown=120):
    """Freeze the committed convergence PRODUCT from the per-seed logbooks (the display contract).

    For every (experiment, seed) parquet: best-so-far NRMSE per generation, mapped to model
    evaluations = (gen+1)*lam, then log-spaced-downsampled to <= `ndown` points (the figure is
    log-log, so this is visually lossless) — small enough to commit as CSV while the heavy parquets
    stay gitignored. Output (cols: experiment, seed, evaluations, best_nrmse):
        cmaes/cmaes_convergence_traces_l{lam}.csv
    """
    wf = ("Weighted_fitness", "Weighted_fitness")
    rows = []
    for exp in experiments:
        for p in sorted(CMA_SEED_DIR.glob(f"ga_logbook_{exp}_lambda{lam}_seed*.parquet")):
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
    out = PRODUCTS / f"cmaes_convergence_traces_l{lam}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"froze convergence traces -> {out.name} ({len(rows)} rows, "
          f"{len({(r['experiment'], r['seed']) for r in rows})} traces)", flush=True)
    return out


def run_experiment(exp, params, stations_meta, forcing, client, lam, seeds, ngen):
    observations = ga.build_observations(exp, stations_meta)
    obs_names = [o.name for o in observations]
    fg_set = FunctionalGroupSet(ga.functional_groups(params["model_parameters"]["bounds"]))
    nb = fg_set.unique_functional_groups_parameters_ordered()
    names = list(nb.keys())
    lo = np.array([p.lower_bound for p in nb.values()], float)
    hi = np.array([p.upper_bound for p in nb.values()], float)
    cost = CostFunction(
        configuration_generator=NoTransportConfigurationGenerator(model_class=NoTransportSpaceOptimizedLightModel),
        functional_groups=fg_set, forcing=forcing, kernel=KernelParameter(biomass_solver="implicit"),
        observations=observations, processor=TimeSeriesScoreProcessor(comparator=nrmse_std_comparator),
    )
    # scatter big read-only data to workers ONCE (broadcast); avoids per-generation re-transmit ->
    # memory growth + leaked semaphores. Mirrors GeneticAlgorithmFactory.create_distributed.
    cost.forcing = client.scatter(cost.forcing, broadcast=True)
    for _name in list(cost.observations):
        cost.observations[_name] = client.scatter(cost.observations[_name], broadcast=True)
    evalstrat = DistributedEvaluation(cost, client)

    rows, t0 = [], time.time()
    for seed in seeds:
        out = CMA_SEED_DIR / f"ga_logbook_{exp}_lambda{lam}_seed{seed}.parquet"  # namespaced by lambda
        if out.exists():                              # per-seed RESUME (robust restart signal)
            nrmse, prm, stop_gen = best_of_logbook(out)
            stop_reason = "reuse"
        else:
            nrmse, prm, df, stop_reason = run_seed(seed, names, lo, hi, obs_names, evalstrat, lam, ngen)
            df.to_parquet(out)
            stop_gen = int(df.index.get_level_values("Generation").max())
        cap = stop_gen >= ngen - 1
        rows.append({"experiment": exp, "seed": seed, "lambda": lam, "best_nrmse": nrmse,
                     "stop_gen": stop_gen, "cap_hit": cap, **prm})
        print(f"    {exp:14s} seed {seed:3d} (λ={lam}) | NRMSE={nrmse:.4f} | stop_gen={stop_gen}"
              + (" CAP!" if cap else "") + f" | stop={stop_reason}", flush=True)
    # copy the best seed's logbook to the canonical name read by the figure scripts
    best = min(rows, key=lambda r: r["best_nrmse"])
    shutil.copyfile(CMA_SEED_DIR / f"ga_logbook_{exp}_lambda{lam}_seed{best['seed']}.parquet",
                    CMA_DIR / f"ga_logbook_{exp}.parquet")
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
    ap.add_argument("--n-seeds", type=int, default=100, help="run seeds 0..n-1 (default 100)")
    ap.add_argument("--seeds", default=None, help="explicit comma-separated seeds (overrides --n-seeds)")
    ap.add_argument("--ngen", type=int, default=1000,
                    help="iteration-cap backstop (default 1000); convergence is sigma<1e-3, so the cap should NOT bind")
    ap.add_argument("--workers", type=int, default=8, help="process workers, single station (default 8)")
    ap.add_argument("--memory-limit", default="4GB", help="per-worker memory, single station (default 4GB)")
    ap.add_argument("--merged-workers", type=int, default=6, help="process workers for MERGED (default 6)")
    ap.add_argument("--merged-mem", default="6GB", help="per-worker memory for MERGED (default 6GB)")
    ap.add_argument("--freeze", action="store_true",
                    help="rebuild the committed products (convergence-traces CSV) from existing per-seed logbooks; no optimisation")
    args = ap.parse_args()
    experiments = [e.strip() for e in args.experiments.split(",") if e.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()] if args.seeds else list(range(args.n_seeds))
    lam = args.lam
    NGEN = args.ngen                                          # used by run_seed (module global)
    results_csv = PRODUCTS / f"cmaes_seed_ensemble_l{lam}.csv"   # committed product, namespaced by lambda

    PRODUCTS.mkdir(parents=True, exist_ok=True)
    CMA_DIR.mkdir(parents=True, exist_ok=True)
    CMA_SEED_DIR.mkdir(parents=True, exist_ok=True)
    if args.freeze:                                          # rebuild committed products only (no runs)
        build_convergence_traces(lam, experiments)
        return
    params = yaml.safe_load(open(ga.ROOT / "parameters.yaml"))
    stations_meta = json.load(open(ga.DATA_DIR / "stations_coords.json"))
    forcing = ga.build_forcing()
    print(f"CMA-ES ensemble: λ={lam} | NGEN cap={NGEN} | {len(seeds)} seeds | experiments={experiments}", flush=True)
    print(f"  per-seed resume on logbooks/cmaes/seeds/ (lambda-namespaced) | CSV -> {results_csv.name}", flush=True)

    all_rows = []
    for exp in experiments:  # fresh client PER EXPERIMENT (releases workers + scattered data between exps)
        client = (Client(n_workers=args.merged_workers, threads_per_worker=1, memory_limit=args.merged_mem)
                  if exp == "MERGED"
                  else Client(n_workers=args.workers, threads_per_worker=1, memory_limit=args.memory_limit))
        try:
            all_rows.extend(run_experiment(exp, params, stations_meta, forcing, client, lam, seeds, NGEN))
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
          + ("(OK — convergence governs)" if total_cap == 0 else "(some capped → raise --ngen)"), flush=True)
    build_convergence_traces(lam, experiments)               # freeze the committed convergence product


if __name__ == "__main__":
    main()
