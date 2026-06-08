"""bats_real_taufixed.py — REHEARSAL: BATS real-obs optimisation with tau_r FIXED.

Tests the advisor's proposal: optimise SeapoPym against the REAL BATS in-situ
biomass, fixing the recruitment parameters (tr_0, gamma_tr) to their literature
reference values (justified by the twin experiments + Sobol showing low
sensitivity / non-identifiability of tau_r), and freeing only E, lambda_0,
gamma_lambda. Question: does the fit still hold once tau_r can no longer absorb
the seasonal-phase offset?

Framework note: a functional-group field passed as a plain scalar is held FIXED;
only `Parameter(...)` fields are optimised (base_functional_group.py).

Config = frozen production GA: SBX eta_c=15, pop 256, pure-Sobol init, tournament 2,
early-stopping (patience 25, tol 1e-3, cap 250). NRMSE cost against real obs.

Outputs: experiment/figures/bats_taufixed_climatology.png ;
         logbook data/ga_logbook_BATS_real_taufixed.parquet ;
    bats_taufixed_climatology.png  — obs vs reference vs tau-fixed-opt vs all-free-opt
    + printed NRMSE table (GA cost at obs points, and monthly-climatology RMSE).
"""
from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import yaml
from dask.distributed import Client
from scipy.stats import qmc

from seapopym.configuration.no_transport import (
    ForcingParameter, ForcingUnit, FunctionalGroupParameter, FunctionalGroupUnit,
    FunctionalTypeParameter, KernelParameter, MigratoryTypeParameter, NoTransportConfiguration,
)
from seapopym.model.no_transport_model import NoTransportSpaceOptimizedLightModel
from seapopym_optimization.algorithm.genetic_algorithm import GeneticAlgorithmParameters
from seapopym_optimization.algorithm.genetic_algorithm.factory import GeneticAlgorithmFactory
from seapopym_optimization.algorithm.genetic_algorithm.logbook import Logbook
from seapopym_optimization.configuration_generator import NoTransportConfigurationGenerator
from seapopym_optimization.cost_function import CostFunction, TimeSeriesScoreProcessor, nrmse_std_comparator
from seapopym_optimization.functional_group import FunctionalGroupSet, NoTransportFunctionalGroup, Parameter
from seapopym_optimization.functional_group.parameter_initialization import random_uniform_exclusive
from seapopym_optimization.observations import DayCycle, TimeSeriesObservation

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIG = ROOT / "experiment" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
WF = ("Weighted_fitness", "Weighted_fitness")
BATS_LAT = 32.0
PKEYS = ["energy_transfert", "tr_0", "gamma_tr", "lambda_temperature_0", "gamma_lambda_temperature"]

CFG = dict(cx_eta=15.0, pop=256, ngen_cap=250, patience=25, tol=1e-3, min_gen=20)

with open(ROOT / "parameters.yaml") as f:
    P = yaml.safe_load(f)
REF = P["model_parameters"]["reference"]
BOUNDS = P["model_parameters"]["bounds"]
GS = P["global_simulation"]
T_START, T_END = GS["start_date"], "2019-12-31"


# --------------------------------------------------------------------------- #
def build_forcing():
    raw = xr.open_zarr(DATA / "stations.zarr").sel(time=slice(T_START, T_END)).load()
    ordered = raw.isel(station=raw.station_lat.argsort().values)
    y = ordered.station_lat.values.astype(np.float32)
    x = np.array([0.0], dtype=np.float32)

    def grid(da):
        arr = da.values.T[:, :, np.newaxis]
        return xr.DataArray(arr, dims=("T", "Y", "X"),
                            coords={"T": ordered.time.values, "Y": ("Y", y), "X": ("X", x)})

    temp = grid(ordered.temperature).expand_dims(Z=[0], axis=1)
    npp = grid(ordered.npp)
    for c, a in {"T": "T", "Z": "Z", "Y": "Y", "X": "X"}.items():
        if c in temp.coords:
            temp[c].attrs["axis"] = a
        if c in npp.coords:
            npp[c].attrs["axis"] = a
    temp.attrs["units"] = "degC"
    npp.attrs["units"] = "mg/m^2/day"
    ic = xr.open_zarr(DATA / "initial_conditions.zarr").load()
    return ForcingParameter(
        temperature=ForcingUnit(forcing=temp), primary_production=ForcingUnit(forcing=npp),
        initial_condition_biomass=ForcingUnit(forcing=ic.biomass),
        initial_condition_production=ForcingUnit(forcing=ic.preproduction),
    )


def build_obs():
    o = xr.open_zarr(DATA / "bats_real_observations.zarr")["observed_biomass"]
    obs = o.sel(Y=BATS_LAT, X=0, method="nearest").assign_coords(Z=0).load()
    for c, a in {"T": "T", "Z": "Z", "Y": "Y", "X": "X"}.items():
        if c in obs.coords:
            obs[c].attrs["axis"] = a
    return [TimeSeriesObservation(name="BATS", observation=obs, observation_type=DayCycle.DAY)]


def fg_taufixed():
    """E, lambda_0, gamma_lambda free (Parameter); tr_0 & gamma_tr FIXED to reference (scalar)."""
    return [NoTransportFunctionalGroup(
        name="zooplankton", day_layer=0, night_layer=0,
        energy_transfert=Parameter("energy_transfert", *BOUNDS["energy_transfert"], init_method=random_uniform_exclusive),
        lambda_temperature_0=Parameter("lambda_temperature_0", *BOUNDS["lambda_temperature_0"], init_method=random_uniform_exclusive),
        gamma_lambda_temperature=Parameter("gamma_lambda_temperature", *BOUNDS["gamma_lambda_temperature"], init_method=random_uniform_exclusive),
        tr_0=REF["tr_0"],            # FIXED (scalar)
        gamma_tr=REF["gamma_tr"],    # FIXED (scalar)
    )]


def sobol_init(fg_set, pop, fitness_names):
    nb = fg_set.unique_functional_groups_parameters_ordered()
    names = list(nb.keys())
    lo = np.array([p.lower_bound for p in nb.values()])
    hi = np.array([p.upper_bound for p in nb.values()])
    m = int(round(np.log2(pop)))
    unit = qmc.Sobol(d=len(names), scramble=True).random_base2(m=m)
    samples = pd.DataFrame(qmc.scale(unit, lo, hi), columns=names)
    return Logbook.from_array(generation=[0] * len(samples), is_from_previous_generation=[False] * len(samples),
                              individual=samples.to_numpy(), parameter_names=names, fitness_names=fitness_names)


def optimize_es(ga, patience, tol, min_gen):
    gen_start, pop = ga._initialization()
    best, no_imp, stop = np.inf, 0, ga.meta_parameter.NGEN - 1
    for gen in range(gen_start, ga.meta_parameter.NGEN):
        off = ga.toolbox.select(pop, ga.meta_parameter.POP_SIZE)
        off = ga.meta_parameter.variation(off, ga.toolbox, ga.meta_parameter.CXPB, ga.meta_parameter.MUTPB)
        ga.update_logbook(ga._evaluate(off, gen))
        pop[:] = off
        cur = float(-ga.logbook[WF].max())
        imp = (best - cur) / best if np.isfinite(best) and best > 0 else 1.0
        no_imp = 0 if imp > tol else no_imp + 1
        best = min(best, cur)
        if gen >= min_gen and no_imp >= patience:
            stop = gen
            break
    return stop, best


# --------------------------------------------------------------------------- #
# post-hoc model run for the climatology
_FORCING_GRID = None


def run_model(params):
    raw = xr.open_zarr(DATA / "stations.zarr").sel(time=slice(T_START, T_END)).load()
    ordered = raw.isel(station=raw.station_lat.argsort().values)
    y = ordered.station_lat.values.astype(np.float32)
    x = np.array([0.0], dtype=np.float32)

    def grid(da):
        arr = da.values.T[:, :, np.newaxis]
        return xr.DataArray(arr, dims=("T", "Y", "X"),
                            coords={"T": ordered.time.values, "Y": ("Y", y), "X": ("X", x)})

    temp = grid(ordered.temperature).expand_dims(Z=[0], axis=1)
    npp = grid(ordered.npp)
    for c, a in {"T": "T", "Z": "Z", "Y": "Y", "X": "X"}.items():
        if c in temp.coords:
            temp[c].attrs["axis"] = a
        if c in npp.coords:
            npp[c].attrs["axis"] = a
    temp.attrs["units"] = "degC"
    npp.attrs["units"] = "mg/m^2/day"
    ic = xr.open_zarr(DATA / "initial_conditions.zarr").load()
    fg = FunctionalGroupParameter(functional_group=[FunctionalGroupUnit(
        name="zooplankton", energy_transfert=params["energy_transfert"],
        functional_type=FunctionalTypeParameter(
            lambda_temperature_0=params["lambda_temperature_0"], gamma_lambda_temperature=params["gamma_lambda_temperature"],
            tr_0=params["tr_0"], gamma_tr=params["gamma_tr"]),
        migratory_type=MigratoryTypeParameter(day_layer=0, night_layer=0))])
    config = NoTransportConfiguration(
        forcing=ForcingParameter(temperature=ForcingUnit(forcing=temp), primary_production=ForcingUnit(forcing=npp),
                                 initial_condition_biomass=ForcingUnit(forcing=ic.biomass),
                                 initial_condition_production=ForcingUnit(forcing=ic.preproduction)),
        functional_group=fg, kernel=KernelParameter())
    with NoTransportSpaceOptimizedLightModel.from_configuration(configuration=config) as model:
        model.run()
        model.state.compute()
        b = model.state.biomass.sel(Y=BATS_LAT, X=0, functional_group=0, method="nearest", drop=True).load()
    return b


def monthly_clim(da):
    return da.groupby(da["T"].dt.month).mean()


def main():
    client = Client(n_workers=8, threads_per_worker=1, memory_limit="4GB")
    print(client.dashboard_link, flush=True)
    forcing = build_forcing()
    obs = build_obs()
    fg_set = FunctionalGroupSet(fg_taufixed())
    free_names = list(fg_set.unique_functional_groups_parameters_ordered().keys())
    print(f"FREE params: {free_names} | FIXED: tr_0={REF['tr_0']}, gamma_tr={REF['gamma_tr']}", flush=True)

    logbook = sobol_init(fg_set, CFG["pop"], [o.name for o in obs])
    out = DATA / "ga_logbook_BATS_real_taufixed.parquet"
    out.parent.mkdir(exist_ok=True)
    if out.exists():
        out.unlink()
    cost = CostFunction(
        configuration_generator=NoTransportConfigurationGenerator(model_class=NoTransportSpaceOptimizedLightModel),
        functional_groups=fg_set, forcing=forcing, kernel=KernelParameter(),
        observations=obs, processor=TimeSeriesScoreProcessor(comparator=nrmse_std_comparator))
    ga_params = GeneticAlgorithmParameters(
        MUTPB=1.0, INDPB=1 / 3, ETA=20.0, CXPB=P["genetic_algorithm"]["cxpb"],
        NGEN=CFG["ngen_cap"], POP_SIZE=CFG["pop"], cost_function_weight=(-1.0,),
        TOURNSIZE=2, CX_METHOD="sbx", CX_ETA=CFG["cx_eta"])
    ga = GeneticAlgorithmFactory.create_distributed(meta_parameter=ga_params, cost_function=cost,
                                                    client=client, logbook=logbook, save=out)
    print("Optimising BATS (tau_r fixed) ...", flush=True)
    t0 = time.time()
    stop, best_nrmse = optimize_es(ga, CFG["patience"], CFG["tol"], CFG["min_gen"])
    client.close()
    df = pd.read_parquet(out)
    best = df.loc[df[WF].idxmax(), "Parametre"].to_dict()
    taufixed = {k: float(best.get(k, REF[k])) for k in PKEYS}
    taufixed["tr_0"], taufixed["gamma_tr"] = REF["tr_0"], REF["gamma_tr"]
    print(f"done {(time.time()-t0)/60:.1f} min | stop_gen={stop} | GA NRMSE={best_nrmse:.4f}", flush=True)
    print("tau-fixed best:", {k: round(v, 5) for k, v in taufixed.items()}, flush=True)

    # all-free real optimisation (existing) for comparison
    allfree = pd.read_parquet(DATA / "ga_logbook_BATS_validation_batsreal.parquet")
    allfree = allfree.loc[allfree[WF].idxmax(), "Parametre"].to_dict()
    allfree = {k: float(allfree[k]) for k in PKEYS}

    # --- model runs + climatology ---
    obs_da = obs[0].observation
    runs = {"reference": REF, "tau-fixed opt": taufixed, "all-free opt": allfree}
    series = {k: run_model(v) for k, v in runs.items()}
    o_clim = monthly_clim(obs_da)
    o_std = obs_da.groupby(obs_da["T"].dt.month).std()

    def nrmse_pts(model_b):
        m = model_b.sel(T=obs_da["T"], method="nearest").values
        y = obs_da.values
        ok = np.isfinite(m) & np.isfinite(y)
        return float(np.sqrt(np.mean((m[ok] - y[ok]) ** 2)) / np.std(y[ok]))

    print(f"\n{'run':16s}{'NRMSE@obs':>12s}")
    for k, b in series.items():
        print(f"{k:16s}{nrmse_pts(b):12.4f}")

    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=200)
    months = np.arange(1, 13)
    ax.fill_between(months, (o_clim - o_std).values, (o_clim + o_std).values, color="green", alpha=0.15)
    ax.plot(months, o_clim.values, "-o", color="green", lw=2, label="In-situ obs (BATS)")
    styles = {"reference": ("--s", "#d62728"), "tau-fixed opt": (":^", "#9467bd"), "all-free opt": ("-.D", "#1f77b4")}
    for k, b in series.items():
        c = monthly_clim(b)
        st, col = styles[k]
        ax.plot(months, c.values, st, color=col, lw=1.6, label=f"{k} (NRMSE={nrmse_pts(b):.2f})")
    ax.set_xticks(months)
    ax.set_xticklabels(list("JFMAMJJASOND"))
    ax.set_xlabel("Month")
    ax.set_ylabel(r"Zooplankton biomass (g C m$^{-2}$)")
    ax.set_title("BATS monthly climatology — tau_r FIXED vs all-free optimisation", fontweight="bold", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    plt.tight_layout()
    p = FIG / "bats_taufixed_climatology.png"
    fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"\nSaved {p}", flush=True)


if __name__ == "__main__":
    main()
