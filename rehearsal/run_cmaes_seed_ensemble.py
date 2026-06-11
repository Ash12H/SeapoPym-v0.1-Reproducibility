"""rehearsal/run_cmaes_seed_ensemble.py — CMA-ES CROSS-CHECK (multi-start, DEAP default lambda).

Independent robustness check of the GA, NOT the headline method: do GA and CMA-ES agree on the
recovered parameters / the regime-dependent reliability story? CMA-ES is a different optimiser
family (adapts a covariance matrix + step size), so agreement is strong evidence the findings are
about the problem, not the optimiser.

Large-scale config (defaults): lambda = DEAP default int(4+3*ln(N)) = 8, 100 fully-random restarts
per experiment (seeds 0..99, no privileged centre start). Convergence is sigma<1e-3 (or ill-conditioned
covariance); NGEN is a high backstop (default 1000) that should NOT bind — a cap_hit flag + count
verify this. Same objective as the GA (mean station NRMSE via CostFunction + DistributedEvaluation);
params normalised to [0,1]^5 (GA bounds), clipped + nan/inf-safe. Compare on PARAMETER RECOVERY vs
reference, not on NRMSE (low NRMSE on warm stations is equifinality, not recovery).

RESUMABLE per seed: each seed's full trajectory is written to a lambda-namespaced logbook; a restart
reuses any seed whose logbook already exists (so a crash loses at most the in-flight seed). Outputs
are ISOLATED (never clobber the GA logbooks) and namespaced by lambda (never clobber another config):
    logbooks/cmaes/seeds/ga_logbook_{exp}_lambda{L}_seed{S}.parquet   per-seed trajectory (resume unit)
    logbooks/cmaes/ga_logbook_{exp}.parquet                           best seed (read by figure scripts)
    logbooks/cmaes_seed_ensemble_l{L}.csv                             all (exp, seed) results + cap_hit
Resumable at the experiment level (marker tag "cmaes_ms").

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
from dask.distributed import Client
from deap import base, cma, creator

import run_ga_production as ga
from seapopym.configuration.no_transport import KernelParameter
from seapopym.model.no_transport_model import NoTransportSpaceOptimizedLightModel
from seapopym_optimization.algorithm.genetic_algorithm.evaluation_strategies import DistributedEvaluation
from seapopym_optimization.configuration_generator import NoTransportConfigurationGenerator
from seapopym_optimization.cost_function import CostFunction, TimeSeriesScoreProcessor, nrmse_std_comparator
from seapopym_optimization.functional_group import FunctionalGroupSet

EXPERIMENTS = ["BARENTS", "PAPA", "Bay_of_Biscay", "BATS", "Canaries", "HOT", "MERGED"]
SIGMA0, NGEN = 0.30, 1000                            # NGEN is a backstop; convergence is sigma<1e-3 (set by main)
CMA_DIR = ga.ROOT / "rehearsal" / "cmaes"            # dedicated production-results folder (the 700 optimisations)
CMA_SEED_DIR = CMA_DIR / "seeds"                     # per-seed trajectories (resume unit), lambda-namespaced
REF = yaml.safe_load(open(ga.ROOT / "parameters.yaml"))["model_parameters"]["reference"]


def cmaes_lambda(n_params: int) -> int:
    """DEAP's default CMA-ES population size, lambda = int(4 + 3*ln(N))."""
    return int(4 + 3 * np.log(n_params))

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)


def run_seed(seed, names, lo, hi, obs_names, evalstrat, lam):
    def denorm(xn):
        xn = np.nan_to_num(np.asarray(xn, float), nan=0.5, posinf=1.0, neginf=0.0)
        return (lo + np.clip(xn, 0.0, 1.0) * (hi - lo)).tolist()

    np.random.seed(seed)
    # Every seed is a fresh random restart in [0,1]^5 — no privileged centre start, so this mirrors
    # the GA ensemble (10 random Sobol inits) and characterises run-to-run reliability the same way.
    start = list(np.random.uniform(0.0, 1.0, len(names)))
    strat = cma.Strategy(centroid=start, sigma=SIGMA0, lambda_=lam)
    tb = base.Toolbox()
    tb.register("generate", strat.generate, creator.Individual)
    tb.register("update", strat.update)
    rows, index, best = [], [], np.inf
    for gen in range(NGEN):
        pop = tb.generate()
        reals = [denorm(ind) for ind in pop]
        fits = evalstrat.evaluate(reals)
        costs = [float(np.mean(np.asarray(f, float))) for f in fits]
        for ind, c in zip(pop, costs):
            ind.fitness.values = (c,)
        tb.update(pop)
        for i, (real, f, c) in enumerate(zip(reals, fits, costs)):
            index.append((gen, False, i))
            row = {("Parametre", p): real[j] for j, p in enumerate(names)}
            row.update({("Fitness", o): float(np.asarray(f, float)[k]) for k, o in enumerate(obs_names)})
            row[("Weighted_fitness", "Weighted_fitness")] = -c
            rows.append(row)
        best = min(best, min(costs))
        cov_bad = (not np.all(np.isfinite(strat.C))) or (np.linalg.cond(strat.C) > 1e12)
        if strat.sigma < 1e-3 or cov_bad:
            break
    df = pd.DataFrame(rows, index=pd.MultiIndex.from_tuples(
        index, names=["Generation", "Is_From_Previous_Generation", "Individual"]))
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    bx = df.loc[df[("Weighted_fitness", "Weighted_fitness")].idxmax(), "Parametre"].to_dict()
    return best, {k: float(bx[k]) for k in ga.PARAM_KEYS}, df


def best_of_logbook(out):
    """(best NRMSE, best params, stop_gen) from an existing per-seed logbook — for resume."""
    df = pd.read_parquet(out)
    wf = df[("Weighted_fitness", "Weighted_fitness")]
    b = df.loc[wf.idxmax(), "Parametre"].to_dict()
    return float(-wf.max()), {k: float(b[k]) for k in ga.PARAM_KEYS}, int(df.index.get_level_values("Generation").max())


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
            reused = True
        else:
            nrmse, prm, df = run_seed(seed, names, lo, hi, obs_names, evalstrat, lam)
            df.to_parquet(out)
            stop_gen, reused = int(df.index.get_level_values("Generation").max()), False
        cap = stop_gen >= ngen - 1
        rows.append({"experiment": exp, "seed": seed, "lambda": lam, "best_nrmse": nrmse,
                     "stop_gen": stop_gen, "cap_hit": cap, **prm})
        print(f"    {exp:14s} seed {seed:3d} (λ={lam}) | NRMSE={nrmse:.4f} | stop_gen={stop_gen}"
              + (" CAP!" if cap else "") + (" [reuse]" if reused else ""), flush=True)
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
    args = ap.parse_args()
    experiments = [e.strip() for e in args.experiments.split(",") if e.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()] if args.seeds else list(range(args.n_seeds))
    lam = args.lam
    NGEN = args.ngen                                          # used by run_seed (module global)
    results_csv = CMA_DIR / f"cmaes_seed_ensemble_l{lam}.csv"   # in the cmaes/ folder, namespaced by lambda

    CMA_DIR.mkdir(parents=True, exist_ok=True)
    CMA_SEED_DIR.mkdir(parents=True, exist_ok=True)
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


if __name__ == "__main__":
    main()
