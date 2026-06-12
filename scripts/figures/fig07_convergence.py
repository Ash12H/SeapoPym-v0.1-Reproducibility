"""Figure 7 — CMA-ES convergence: best-of-ensemble trajectory per experiment.

For each experiment (MERGED + 6 stations), the best of the 20 seeded restarts: its best-so-far cost
vs model evaluations. log x-axis (the descent spans 8 -> ~10^4 evaluations); LINEAR y so the near-zero
region is not exaggerated. Every experiment converges to the twin global optimum (cost = 0) except
HOT, whose best restart floors above zero (its information-limited structural minimum). Lines are
layered earliest-to-stop (fewest evaluations to termination) in front. The per-seed spread is in the
convergence-dispersion figure.

Reads ONLY the committed convergence-traces product for the production cost (paths.PRODUCTION_METRIC),
frozen by scripts/experiments/run_cmaes_seed_ensemble.py. Use --metric to render another cost's run.
Output : figures/Figure_7.{pdf,png}
Run    : .venv/bin/python scripts/figures/fig07_convergence.py [--metric nrmse_mean]
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

from seapopym_repro import figstyle as fs, paths

ap = argparse.ArgumentParser()
ap.add_argument("--metric", default=paths.PRODUCTION_METRIC, help="cost metric whose frozen run to plot")
METRIC = ap.parse_args().metric
YLABEL = {"nrmse_std": "best-so-far NRMSE (std-norm.)", "nrmse_mean": "best-so-far NRMSE (mean-norm.)",
          "rmse": "best-so-far RMSE", "mae": "best-so-far MAE", "nmae": "best-so-far nMAE"}.get(METRIC, "best-so-far cost")

traces = pd.read_csv(paths.cmaes_product("convergence_traces", METRIC))
ORDER = fs.order(traces.experiment.unique())   # MERGED + 6 stations, cold -> warm


def best_trajectory(exp):
    sub = traces[traces.experiment == exp]
    best_seed = sub.groupby("seed").best_nrmse.min().idxmin()   # seed reaching the lowest NRMSE
    t = sub[sub.seed == best_seed].sort_values("evaluations")
    return t.evaluations.to_numpy(), t.best_nrmse.to_numpy()


trajs = {e: best_trajectory(e) for e in ORDER}


def stop_eval(e):
    """Evaluations at which CMA-ES terminated for this experiment (the best trajectory's last point;
    trajectories are sorted by evaluations, so ev[-1] is the stop). Earlier stop -> drawn in front."""
    ev, _ = trajs[e]
    return ev[-1]


fig, ax = plt.subplots(figsize=(0.78 * fs.WIDTH_FULL, 0.52 * fs.WIDTH_FULL))
for rank, e in enumerate(sorted(ORDER, key=stop_eval)):   # earliest stop -> foreground
    ev, b = trajs[e]
    ax.plot(ev, b, color=fs.color(e), lw=2.0 if e == "MERGED" else 1.8,
            zorder=5 + len(ORDER) - rank, solid_capstyle="round")

ax.set_xscale("log")
ax.set_ylim(-0.1, None)   # small negative floor lifts the converged plateaus off the bottom frame
ax.set_xlabel("model evaluations")
ax.set_ylabel(YLABEL)
ax.grid(True, which="both", alpha=0.22, linewidth=0.5)

handles = [Line2D([0], [0], color=fs.color(e), lw=2.2, label=fs.label(e)) for e in ORDER]
ax.legend(handles=handles, loc="upper right", ncol=2, fontsize=7.5, frameon=True)
fig.tight_layout()
fs.save(fig, "Figure_7", subdir=None)
