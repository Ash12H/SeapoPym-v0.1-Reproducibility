"""run_ga_production.py — core GA twin-experiment module + single-run driver.

Shared library for the rehearsal GA pipeline. The reliability ensemble
(run_seed_ensemble.py) and the CMA-ES cross-check (run_cmaes_seed_ensemble.py)
import the helpers defined here (build_forcing, build_observations,
functional_groups, sobol_init_logbook, optimize_with_early_stopping, CFG, ...).

ONE frozen config (no test/validation/production modes — the 0D twin is already
fast enough that a single config suffices):
    metric      : NRMSE (nrmse_std)               -- the published, normalised cost
    crossover   : SBX (cxSimulatedBinaryBounded)  -- canonical partner of the
                  polynomial-bounded mutation; eta_c = 15
    population  : 64 (= 2^6)                       -- ~13x the 5 params; keeps the
                  Sobol gen-0 balance (power of 2). Small enough to run a seeded
                  ensemble cheaply, large enough for GA diversity (cf. CMA-ES
                  default lambda=8, too small for a memoryless GA).
    stopping    : patience + tolerance on the best-so-far NRMSE
                  (patience=25, tol=1e-3 = 0.1% relative improvement, min_gen=20,
                   cap=250 as a backstop)
    experiments : BARENTS, PAPA, Bay_of_Biscay, BATS, Canaries, HOT, MERGED

Dask runs THREADED (processes=False, n_workers=1): a single in-process scheduler
+ a one-shot broadcast scatter. This sidesteps the multiprocessing semaphore leak
AND the per-process OOM that killed the previous process-based runs ~1 experiment
in (each 0D eval is tiny, so threading avoids serialisation overhead too).

This single-run driver writes the canonical ga_logbook_{exp}.parquet that the
figures (fig07/08/09) read. The reliability spread across seeds is the job of
run_seed_ensemble.py, which writes per-seed logbooks to logbooks/seeds/.
MERGED runs LAST so a failure there does not lose the single-station runs.

Run (default = all 7, seed 0):
    .venv/bin/python rehearsal/run_ga_production.py
    .venv/bin/python rehearsal/run_ga_production.py --experiments BARENTS     # sanity
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml
from dask.distributed import Client

from seapopym.configuration.no_transport import ForcingParameter, ForcingUnit, KernelParameter
from seapopym.model.no_transport_model import NoTransportSpaceOptimizedLightModel
from seapopym_optimization.algorithm.genetic_algorithm import GeneticAlgorithmParameters
from seapopym_optimization.algorithm.genetic_algorithm.factory import GeneticAlgorithmFactory
from seapopym_optimization.algorithm.genetic_algorithm.logbook import Logbook
from seapopym_optimization.configuration_generator import NoTransportConfigurationGenerator
from seapopym_optimization.cost_function import CostFunction, TimeSeriesScoreProcessor, nrmse_std_comparator
from seapopym_optimization.functional_group import FunctionalGroupSet, NoTransportFunctionalGroup, Parameter
from seapopym_optimization.functional_group.parameter_initialization import random_uniform_exclusive
from seapopym_optimization.observations import DayCycle, TimeSeriesObservation
from scipy.stats import qmc

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "rehearsal" / "logbooks"

PARAM_KEYS = ["energy_transfert", "tr_0", "gamma_tr", "lambda_temperature_0", "gamma_lambda_temperature"]
EXPERIMENTS = ["BARENTS", "PAPA", "Bay_of_Biscay", "BATS", "Canaries", "HOT", "MERGED"]
WF = ("Weighted_fitness", "Weighted_fitness")

# Frozen config (single, no modes) ---------------------------------------------
CFG = dict(
    crossover="sbx", cx_eta=15.0,  # eta_c = 15 (more explorative than Deb's 20; pymoo default)
    pop=64, ngen_cap=250,          # gen-0 = a balanced Sobol sample of EXACTLY `pop` points
                                   # (64 = 2^6) -> no member dropped between gen-0 and the
                                   # working population. cap = backstop; early-stopping governs.
    patience=25, tol=1e-3, min_gen=20,
)


# =============================================================================
# Early stopping: replicate GeneticAlgorithm.optimize() + break on patience.
# Monitors the best-so-far NRMSE (= -max Weighted_fitness, monotone). Stop when
# relative improvement < tol for `patience` consecutive generations.
# =============================================================================
def optimize_with_early_stopping(ga, patience, tol, min_gen):
    generation_start, population = ga._initialization()
    best_cost, no_improve, stop_gen = np.inf, 0, ga.meta_parameter.NGEN - 1
    for gen in range(generation_start, ga.meta_parameter.NGEN):
        offspring = ga.toolbox.select(population, ga.meta_parameter.POP_SIZE)
        offspring = ga.meta_parameter.variation(
            offspring, ga.toolbox, ga.meta_parameter.CXPB, ga.meta_parameter.MUTPB
        )
        ga.update_logbook(ga._evaluate(offspring, gen))
        population[:] = offspring
        cur = float(-ga.logbook[WF].max())
        improvement = (best_cost - cur) / best_cost if np.isfinite(best_cost) and best_cost > 0 else 1.0
        no_improve = 0 if improvement > tol else no_improve + 1
        best_cost = min(best_cost, cur)
        if gen >= min_gen and no_improve >= patience:
            stop_gen = gen
            print(f"     early stop @ gen {gen} (best NRMSE={best_cost:.5g})", flush=True)
            break
    return ga.logbook.copy(), stop_gen, best_cost


# =============================================================================
# Pipeline (mirrors scripts/run_optimization.py).
# =============================================================================
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
    ic = xr.open_zarr(DATA_DIR / "initial_conditions.zarr").load()
    return ForcingParameter(
        temperature=ForcingUnit(forcing=temperature),
        primary_production=ForcingUnit(forcing=npp),
        initial_condition_biomass=ForcingUnit(forcing=ic.biomass),
        initial_condition_production=ForcingUnit(forcing=ic.preproduction),
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


def sobol_init_logbook(fg_set, pop, fitness_names, seed=None):
    """Initial population = EXACTLY `pop` balanced Sobol points scaled to the bounds.

    Uses a pure scrambled Sobol sequence (scipy qmc), not SALib's Saltelli N*(D+2)
    design, so gen-0 size == working population (no member dropped at gen-0 -> gen-1).
    pop should be a power of 2 for the Sobol balance property (256 = 2^8, 512 = 2^9).
    `seed` makes the scramble (hence the whole run) reproducible.
    """
    nb = fg_set.unique_functional_groups_parameters_ordered()
    names = list(nb.keys())
    lo = np.array([p.lower_bound for p in nb.values()])
    hi = np.array([p.upper_bound for p in nb.values()])
    sampler = qmc.Sobol(d=len(names), scramble=True, seed=seed)
    m = int(round(np.log2(pop)))
    unit = sampler.random_base2(m=m) if 2 ** m == pop else sampler.random(pop)
    samples = pd.DataFrame(qmc.scale(unit, lo, hi), columns=names)
    return Logbook.from_array(
        generation=[0] * len(samples), is_from_previous_generation=[False] * len(samples),
        individual=samples.to_numpy(), parameter_names=names, fitness_names=fitness_names,
    )


def run_one(experiment, params, stations_meta, forcing, client, biomass_solver="implicit", seed=None):
    if seed is not None:  # reproducible run: seed Sobol init scramble + DEAP operators
        random.seed(seed)
        np.random.seed(seed)
    bounds = params["model_parameters"]["bounds"]
    observations = build_observations(experiment, stations_meta)
    fg_set = FunctionalGroupSet(functional_groups(bounds))
    logbook = sobol_init_logbook(fg_set, CFG["pop"], [o.name for o in observations], seed=seed)
    out = OUT_DIR / f"ga_logbook_{experiment}.parquet"
    if out.exists():
        out.unlink()
    cost = CostFunction(
        configuration_generator=NoTransportConfigurationGenerator(model_class=NoTransportSpaceOptimizedLightModel),
        functional_groups=fg_set, forcing=forcing, kernel=KernelParameter(biomass_solver=biomass_solver),
        observations=observations, processor=TimeSeriesScoreProcessor(comparator=nrmse_std_comparator),
    )
    ga_params = GeneticAlgorithmParameters(
        MUTPB=1.0, INDPB=1 / 5, ETA=20.0, CXPB=params["genetic_algorithm"]["cxpb"],
        NGEN=CFG["ngen_cap"], POP_SIZE=CFG["pop"], cost_function_weight=tuple(-1.0 for _ in observations),
        TOURNSIZE=2,  # binary tournament (citable; Deb et al. 2002)
        CX_METHOD=CFG["crossover"], CX_ETA=CFG["cx_eta"],
    )
    ga = GeneticAlgorithmFactory.create_distributed(
        meta_parameter=ga_params, cost_function=cost, client=client, logbook=logbook, save=out,
    )
    print(f"  -> {experiment:14s} | {len(observations)} obs | SBX pop={CFG['pop']} cap={CFG['ngen_cap']}", flush=True)
    t0 = time.time()
    _, stop_gen, best_cost = optimize_with_early_stopping(ga, CFG["patience"], CFG["tol"], CFG["min_gen"])
    dt = time.time() - t0
    best = pd.read_parquet(out).loc[lambda d: d[WF].idxmax(), "Parametre"].to_dict()
    print(f"     done {dt/60:.1f} min | stop_gen={stop_gen} | NRMSE={best_cost:.5g} -> {out.name}", flush=True)
    return {"experiment": experiment, "stop_gen": stop_gen, "best_nrmse": best_cost,
            "minutes": dt / 60, **{k: float(best[k]) for k in PARAM_KEYS}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiments", default=",".join(EXPERIMENTS),
                    help="comma-separated subset (default: all 7)")
    ap.add_argument("--biomass-solver", choices=["explicit", "implicit"], default="implicit",
                    help="biomass time scheme used by the GA model (default: implicit)")
    ap.add_argument("--pop", type=int, default=None, help=f"override population size (default {CFG['pop']})")
    ap.add_argument("--seed", type=int, default=0,
                    help="seed the Sobol init scramble + DEAP operators -> reproducible run (default 0)")
    ap.add_argument("--no-resume", action="store_true",
                    help="recompute every experiment even if a .done completion marker exists")
    ap.add_argument("--threads", type=int, default=8,
                    help="threads for the in-process Dask scheduler (default 8)")
    args = ap.parse_args()
    experiments = [e.strip() for e in args.experiments.split(",") if e.strip()]

    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(ROOT / "parameters.yaml") as f:
        params = yaml.safe_load(f)
    with open(DATA_DIR / "stations_coords.json") as f:
        stations_meta = json.load(f)

    if args.pop is not None:
        CFG["pop"] = args.pop
    # completion marker encodes pop+seed, so resume never reuses a logbook from a different config
    tag = f"pop{CFG['pop']}_seed{args.seed}"

    forcing = build_forcing()
    # Threaded in-process scheduler (see module docstring): no semaphore leak, no per-worker OOM.
    client = Client(processes=False, n_workers=1, threads_per_worker=args.threads)
    print(f"Dask dashboard: {client.dashboard_link}", flush=True)
    print(f"tag={tag} solver={args.biomass_solver} | {CFG} | experiments={experiments}", flush=True)

    summary = []
    try:
        for exp in experiments:  # MERGED is last in the default order
            out = OUT_DIR / f"ga_logbook_{exp}.parquet"
            done = OUT_DIR / f"ga_logbook_{exp}.done"
            # resume only skips an experiment already completed with THIS exact config (marker = tag)
            if not args.no_resume and out.exists() and done.exists() and done.read_text().strip() == tag:
                d = pd.read_parquet(out)
                best = d.loc[d[WF].idxmax(), "Parametre"].to_dict()
                print(f"  -> {exp:14s} | already complete ({tag}), skipping (NRMSE={float(-d[WF].max()):.5g})", flush=True)
                summary.append({"experiment": exp, "stop_gen": -1, "best_nrmse": float(-d[WF].max()),
                                "minutes": 0.0, **{k: float(best[k]) for k in PARAM_KEYS}})
                continue
            summary.append(run_one(exp, params, stations_meta, forcing, client, args.biomass_solver, args.seed))
            done.write_text(tag)  # mark this logbook as completed with this exact config
    finally:
        client.close()

    df = pd.DataFrame(summary)
    df.to_csv(OUT_DIR / "summary.csv", index=False)
    print("\n===== SUMMARY =====", flush=True)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.5g}"), flush=True)
    print(f"\nLogbooks + summary in {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
