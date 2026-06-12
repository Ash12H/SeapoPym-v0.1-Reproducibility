"""Supplementary figure — convergence dispersion across the seeded CMA-ES restarts.

Per experiment, the distribution of the best-of-run NRMSE reached by each of the N seeds: a boxplot
(quartiles) with the individual seeds overlaid, on a log y-axis (the achieved NRMSE spans several
orders of magnitude). Shows convergence RELIABILITY — stations where every restart reaches ~0 (tight
low box) vs. those with a spread of outcomes — and HOT's elevated structural floor (~0.07-0.15, no
restart reaches 0). Companion to Figure 7 (which shows only the best trajectory per station).

Reads ONLY products/cmaes_seed_ensemble_l8.csv (frozen by run_cmaes_seed_ensemble.py).
Output : figures/convergence_dispersion.{pdf,png}
Run    : .venv/bin/python scripts/figures/fig_convergence_dispersion.py
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

from seapopym_repro import figstyle as fs, paths

FLOOR = 1e-4   # NRMSE below this is "fully converged"; clip so the log axis isn't stretched to noise

d = pd.read_csv(paths.PRODUCTS / "cmaes_seed_ensemble_l8.csv")
ORDER = fs.order(d.experiment.unique())
vals = {e: np.clip(d.loc[d.experiment == e, "best_nrmse"].to_numpy(), FLOOR, None) for e in ORDER}
x = np.arange(1, len(ORDER) + 1)

fig, ax = plt.subplots(figsize=(0.72 * fs.WIDTH_FULL, 0.50 * fs.WIDTH_FULL))
bp = ax.boxplot([vals[e] for e in ORDER], positions=x, widths=0.6, patch_artist=True,
                showfliers=False, medianprops=dict(color="black", lw=1.3),
                whiskerprops=dict(color="0.4"), capprops=dict(color="0.4"), boxprops=dict(linewidth=0.6))
for patch, e in zip(bp["boxes"], ORDER):
    patch.set_facecolor(fs.color(e))
    patch.set_alpha(0.45)
    patch.set_edgecolor("black")

for i, e in zip(x, ORDER):   # overlay the individual seeds
    v = vals[e]
    jit = (np.random.RandomState(0).rand(len(v)) - 0.5) * 0.34
    ax.scatter(i + jit, v, s=16, color=fs.color(e), edgecolor="black", linewidth=0.4, alpha=0.85, zorder=5)

ax.set_yscale("log")
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
ax.set_xticks(x)
ax.set_xticklabels([fs.label(e) for e in ORDER], rotation=45, ha="right")
for tl, e in zip(ax.get_xticklabels(), ORDER):
    tl.set_color(fs.color(e))
ax.set_ylabel("best-of-run NRMSE per seed")
ax.grid(True, axis="y", which="both", alpha=0.25)
fig.tight_layout()
fs.save(fig, "convergence_dispersion", subdir=None)
