"""Figure 7 — CMA-ES convergence: best-of-ensemble trajectory per experiment.

For each experiment (MERGED + 6 stations), the best of the 10 seeded restarts: its best-so-far NRMSE
vs model evaluations. log x-axis (the descent spans 8 -> ~10^4 evaluations); LINEAR y so the near-zero
region is not exaggerated. Every experiment converges to the twin global optimum (NRMSE = 0) except
HOT, whose best restart floors at ~0.07 (its information-limited structural minimum). Lines are layered
fastest-to-converge in front. The per-seed spread is shown in Figure 8 (recovery).

Reads ONLY the committed product products/cmaes_convergence_traces_l8.csv (frozen by
scripts/experiments/run_cmaes_seed_ensemble.py).
Output : figures/Figure_7.{pdf,png}
Run    : .venv/bin/python scripts/figures/fig07_convergence.py
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

from seapopym_repro import figstyle as fs, paths

traces = pd.read_csv(paths.PRODUCTS / "cmaes_convergence_traces_l8.csv")
ORDER = fs.order(traces.experiment.unique())   # MERGED + 6 stations, cold -> warm


def best_trajectory(exp):
    sub = traces[traces.experiment == exp]
    best_seed = sub.groupby("seed").best_nrmse.min().idxmin()   # seed reaching the lowest NRMSE
    t = sub[sub.seed == best_seed].sort_values("evaluations")
    return t.evaluations.to_numpy(), t.best_nrmse.to_numpy()


trajs = {e: best_trajectory(e) for e in ORDER}


def converged_at(e, thresh=0.1):
    """Evaluations to first reach NRMSE <= thresh (best-so-far is monotone, so a clean crossing)."""
    ev, b = trajs[e]
    below = ev[b <= thresh]
    return below[0] if len(below) else float("inf")


fig, ax = plt.subplots(figsize=(0.78 * fs.WIDTH_FULL, 0.52 * fs.WIDTH_FULL))
for rank, e in enumerate(sorted(ORDER, key=converged_at)):   # fastest to converge -> foreground
    ev, b = trajs[e]
    ax.plot(ev, b, color=fs.color(e), lw=2.4 if e == "MERGED" else 1.8,
            zorder=5 + len(ORDER) - rank, solid_capstyle="round")

ax.set_xscale("log")
ax.set_ylim(-0.1, None)   # small negative floor lifts the converged plateaus off the bottom frame
ax.set_xlabel("model evaluations")
ax.set_ylabel("best-so-far NRMSE")
ax.grid(True, which="both", alpha=0.22, linewidth=0.5)

handles = [Line2D([0], [0], color=fs.color(e), lw=2.2, label=fs.label(e)) for e in ORDER]
ax.legend(handles=handles, loc="upper right", ncol=2, fontsize=7.5,
          frameon=True, framealpha=0.95, edgecolor="0.7")
fig.tight_layout()
fs.save(fig, "Figure_7", subdir=None)
