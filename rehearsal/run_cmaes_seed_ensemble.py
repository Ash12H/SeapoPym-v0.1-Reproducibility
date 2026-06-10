"""rehearsal/run_cmaes_seed_ensemble.py — CMA-ES CROSS-CHECK (multi-start, DEAP default lambda).

Independent robustness check of the GA, NOT the headline method: do GA and CMA-ES agree on the
recovered parameters / the regime-dependent reliability story? CMA-ES is a different optimiser
family (adapts a covariance matrix + step size), so agreement is strong evidence the findings are
about the problem, not the optimiser.

Multi-start over 10 fully-random restarts per experiment (strict peer of the GA ensemble: same
10 seeds, no privileged centre start; keep the best per experiment for the logbook, report the
spread over all 10) at lambda = DEAP's default int(4 + 3*ln(N)) = 8 for the 5 parameters. Same
objective as the GA (mean station NRMSE via CostFunction + DistributedEvaluation); params
normalised to [0,1]^5 (GA bounds), clipped + nan/inf-safe; robust per-start termination
(sigma<1e-3 OR ill-conditioned cov). Compare on PARAMETER RECOVERY vs reference, not on NRMSE
(low NRMSE on warm stations is equifinality, not recovery).

Outputs are ISOLATED so they never clobber the GA's canonical ga_logbook_{exp}.parquet:
    logbooks/cmaes/ga_logbook_{exp}.parquet   best seed's GA-compatible logbook (cross-check)
    logbooks/cmaes_seed_ensemble.csv          all (exp, seed) results
Resumable at the experiment level (marker tag "cmaes_ms").

Run: .venv/bin/python rehearsal/run_cmaes_seed_ensemble.py
"""

from __future__ import annotations

import json
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
SEEDS = list(range(10))                              # 10 random restarts (strict peer of the GA ensemble)
SIGMA0, NGEN = 0.30, 200
TAG = "cmaes_ms10"   # all-random multi-start, 10 seeds, lambda = DEAP default
CMA_DIR = ga.OUT_DIR / "cmaes"                       # isolated: never overwrites the GA logbooks
CMA_SEED_DIR = CMA_DIR / "seeds"                     # per-seed trajectories (for convergence figures)
RESULTS = ga.OUT_DIR / "cmaes_seed_ensemble.csv"
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


def run_experiment(exp, params, stations_meta, forcing, client):
    observations = ga.build_observations(exp, stations_meta)
    obs_names = [o.name for o in observations]
    fg_set = FunctionalGroupSet(ga.functional_groups(params["model_parameters"]["bounds"]))
    nb = fg_set.unique_functional_groups_parameters_ordered()
    names = list(nb.keys())
    lo = np.array([p.lower_bound for p in nb.values()], float)
    hi = np.array([p.upper_bound for p in nb.values()], float)
    lam = cmaes_lambda(len(names))
    cost = CostFunction(
        configuration_generator=NoTransportConfigurationGenerator(model_class=NoTransportSpaceOptimizedLightModel),
        functional_groups=fg_set, forcing=forcing, kernel=KernelParameter(biomass_solver="implicit"),
        observations=observations, processor=TimeSeriesScoreProcessor(comparator=nrmse_std_comparator),
    )
    # CRITICAL: scatter big data to the workers ONCE (broadcast) so every client.map reuses the same
    # Futures instead of re-transmitting forcing+observations each generation. Skipping this (i.e.
    # passing raw objects to DistributedEvaluation) re-scatters every generation -> memory growth +
    # leaked semaphores -> crash after ~2 experiments. This mirrors GeneticAlgorithmFactory.create_distributed.
    cost.forcing = client.scatter(cost.forcing, broadcast=True)
    for _name in list(cost.observations):
        cost.observations[_name] = client.scatter(cost.observations[_name], broadcast=True)
    evalstrat = DistributedEvaluation(cost, client)

    rows, best_overall, best_df = [], np.inf, None
    t0 = time.time()
    for seed in SEEDS:
        nrmse, prm, df = run_seed(seed, names, lo, hi, obs_names, evalstrat, lam)
        df.to_parquet(CMA_SEED_DIR / f"ga_logbook_{exp}_seed{seed}.parquet")  # full trajectory per seed
        rows.append({"experiment": exp, "seed": seed, "best_nrmse": nrmse, **prm})
        if nrmse < best_overall:
            best_overall, best_df = nrmse, df
        print(f"    {exp:14s} seed {seed} (lambda={lam}) | mean-NRMSE={nrmse:.4f}", flush=True)
    best_df.to_parquet(CMA_DIR / f"ga_logbook_{exp}.parquet")   # best seed (isolated cross-check output)
    (CMA_DIR / f"ga_logbook_{exp}.done").write_text(TAG)
    vals = [r["best_nrmse"] for r in rows]
    print(f"  {exp:14s} | best={min(vals):.4f} median={np.median(vals):.4f} "
          f"max={max(vals):.4f} spread={max(vals)-min(vals):.4f} | {(time.time()-t0)/60:.1f} min", flush=True)
    return rows


def main():
    CMA_DIR.mkdir(parents=True, exist_ok=True)
    CMA_SEED_DIR.mkdir(parents=True, exist_ok=True)
    params = yaml.safe_load(open(ga.ROOT / "parameters.yaml"))
    stations_meta = json.load(open(ga.DATA_DIR / "stations_coords.json"))
    forcing = ga.build_forcing()
    print(f"multi-start CMA-ES cross-check: lambda={cmaes_lambda(5)} (DEAP default, N=5) seeds={SEEDS}", flush=True)
    existing = pd.read_csv(RESULTS) if RESULTS.exists() else None
    all_rows = []
    for exp in EXPERIMENTS:
        done = CMA_DIR / f"ga_logbook_{exp}.done"
        if done.exists() and done.read_text().strip() == TAG:
            print(f"  {exp:14s} | already done (seed ensemble), skipping", flush=True)
            if existing is not None:
                all_rows.extend(existing[existing.experiment == exp].to_dict("records"))
            continue
        # process workers (true parallelism; ~5x faster than threaded), fresh PER EXPERIMENT so the
        # workers + scattered data are released between experiments (this is what avoids the leak that
        # killed the old single-long-lived-client run ~1 exp in). MERGED is memory-heavy -> fewer x more.
        client = (Client(n_workers=6, threads_per_worker=1, memory_limit="6GB") if exp == "MERGED"
                  else Client(n_workers=8, threads_per_worker=1, memory_limit="4GB"))
        try:
            all_rows.extend(run_experiment(exp, params, stations_meta, forcing, client))
            pd.DataFrame(all_rows).to_csv(RESULTS, index=False)   # checkpoint after each experiment
        finally:
            client.close()

    df = pd.DataFrame(all_rows)
    df.to_csv(RESULTS, index=False)
    g = df.groupby("experiment").best_nrmse.agg(["min", "median", "max", "std", "count"])
    print(f"\n===== CMA-ES cross-check (5 seeds, lambda={cmaes_lambda(5)}) — mean-NRMSE per experiment =====", flush=True)
    print(g.to_string(float_format=lambda x: f"{x:.4f}"), flush=True)


if __name__ == "__main__":
    main()
