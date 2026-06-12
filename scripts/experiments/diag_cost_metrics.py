"""Diagnostic (cost-metric test 1) — re-score the best member of each optimisation under several metrics.

Runs the model ONCE at the best-recovered parameters of each experiment (from the seeded ensemble) and
computes, per station, the fit under nrmse_std / nrmse_mean / rmse / mae. Tests whether the std-
normalisation (the current cost) inflates low-variability stations (HOT) while the absolute/relative
error stays small. Serial (no Dask) so it can run alongside an ensemble.

Input  : --csv (the frozen seed-ensemble CSV; default products/cmaes_seed_ensemble_l8.csv) + data/ inputs
Output : products/cost_metric_diagnostic.csv + figures/cost_metric_diagnostic.{pdf,png}
Run    : .venv/bin/python scripts/experiments/diag_cost_metrics.py [--csv PATH]
"""
from __future__ import annotations

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from seapopym.configuration.no_transport import KernelParameter
from seapopym.model.no_transport_model import NoTransportSpaceOptimizedLightModel
from seapopym_optimization.configuration_generator import NoTransportConfigurationGenerator
from seapopym_optimization.cost_function import CostFunction, TimeSeriesScoreProcessor
from seapopym_optimization.functional_group import FunctionalGroupSet

from seapopym_repro import experiment as exp, figstyle as fs, paths
from seapopym_repro.metrics import COMPARATORS

ap = argparse.ArgumentParser()
ap.add_argument("--csv", default=str(paths.PRODUCTS / "cmaes_seed_ensemble_l8.csv"))
args = ap.parse_args()

d = pd.read_csv(args.csv)
bounds = paths.load_params()["model_parameters"]["bounds"]
stations_meta = json.load(open(paths.DATA / "stations_coords.json"))
forcing = exp.build_forcing()
fg_set = FunctionalGroupSet(exp.functional_groups(bounds))
NAMES = list(fg_set.unique_functional_groups_parameters_ordered().keys())   # model arg order


def best_args(e):
    row = d[d.experiment == e].loc[d[d.experiment == e].best_nrmse.idxmin()]
    return np.array([float(row[n]) for n in NAMES])


rows = []
for e in exp.EXPERIMENTS:
    observations = exp.build_observations(e, stations_meta)
    obsd = {o.name: o for o in observations}
    a = best_args(e)
    for mname, comp in COMPARATORS.items():
        cost = CostFunction(
            configuration_generator=NoTransportConfigurationGenerator(model_class=NoTransportSpaceOptimizedLightModel),
            functional_groups=fg_set, forcing=forcing, kernel=KernelParameter(biomass_solver="implicit"),
            observations=observations, processor=TimeSeriesScoreProcessor(comparator=comp))
        scores = cost._cost_function(a, forcing, obsd)   # serial: tuple of per-station scores
        for station, sc in zip(obsd.keys(), scores):
            rows.append(dict(experiment=e, station=station, metric=mname, score=float(sc)))

res = pd.DataFrame(rows)
res.to_csv(paths.PRODUCTS / "cost_metric_diagnostic.csv", index=False)
print("=== single-station best: own-station fit under each metric ===")
single = res[res.experiment == res.station].pivot(index="station", columns="metric", values="score")
print(single.reindex(fs.order(single.index))[["nrmse_std", "nrmse_mean", "rmse", "mae"]].round(4).to_string())

# --- figure: single-station floor under each metric (grouped bars, log y) ---
STATIONS = fs.order([s for s in res.station.unique() if (res.experiment == s).any()])
METRICS = ["nrmse_std", "nrmse_mean", "rmse", "mae"]
MCOL = {"nrmse_std": "#C0392B", "nrmse_mean": "#E67E22", "rmse": "#2E86C1", "mae": "#17A589"}
x = np.arange(len(STATIONS))
w = 0.2
fig, ax = plt.subplots(figsize=(0.78 * fs.WIDTH_FULL, 0.48 * fs.WIDTH_FULL))
for k, m in enumerate(METRICS):
    vals = [float(res[(res.experiment == s) & (res.station == s) & (res.metric == m)].score.iloc[0]) for s in STATIONS]
    ax.bar(x + (k - 1.5) * w, np.clip(vals, 1e-5, None), w, color=MCOL[m], edgecolor="black", linewidth=0.3, label=m)
ax.set_yscale("log")
ax.set_xticks(x)
ax.set_xticklabels([fs.label(s) for s in STATIONS])
for tl, s in zip(ax.get_xticklabels(), STATIONS):
    tl.set_color(fs.color(s))
ax.set_ylabel("best single-station fit (per metric)")
ax.grid(True, axis="y", which="both", alpha=0.25)
ax.legend(ncol=4, fontsize=8, frameon=True, framealpha=0.95, edgecolor="0.7", loc="upper left")
fig.tight_layout()
fs.save(fig, "cost_metric_diagnostic", subdir=None)
