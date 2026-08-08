"""Figure 6: Sobol sensitivity of the biomass magnitude and seasonal timing.

Two rows of descriptors by five columns of parameters. Each panel gives the first-order index S1 and
the total-order index ST at the six stations, ordered from cold to warm, with 95 % bootstrap
confidence intervals. The magnitude is taken as the base-10 logarithm of the mean biomass and the
timing as the day of year of the maximum.

Input  : products/sobol_indices.csv, frozen by scripts/experiments/freeze_sobol_indices.py
Output : figures/Figure_6.pdf and .png
Run    : .venv/bin/python scripts/figures/fig06_sobol.py
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from seapopym_repro import figstyle as fs, paths

CHOSEN_METRICS = ["log10(mean)", "argmax"]
METRIC_LABEL = {"log10(mean)": "Magnitude\n" r"($\log_{10}$ mean biomass)", "argmax": "Peak timing\n(day of year)"}
C_S1, C_ST = fs.BLUE, fs.ORANGE   # S1 vs ST, same blue/orange pair as Figure 5

d = pd.read_csv(paths.PRODUCTS / "sobol_indices.csv")
STATIONS = fs.order(d.station.unique())   # the 6 single stations, coldest -> warmest
PARAMS = fs.PARAM_ORDER
x = np.arange(len(STATIONS))
w = 0.38

fig, axes = plt.subplots(len(CHOSEN_METRICS), len(PARAMS), figsize=(fs.WIDTH_FULL, 0.46 * fs.WIDTH_FULL),
                         sharex=True, sharey=True, squeeze=False)
for r, metric in enumerate(CHOSEN_METRICS):
    for c, p in enumerate(PARAMS):
        ax = axes[r][c]
        sub = d[(d.metric == metric) & (d.param == p)].set_index("station").loc[STATIONS]
        ax.bar(x - w / 2, sub.S1, w, yerr=sub.S1_conf, capsize=2, color=C_S1, edgecolor="k",
               linewidth=0.3, error_kw={"linewidth": 0.6}, label="$S_1$")
        ax.bar(x + w / 2, sub.ST, w, yerr=sub.ST_conf, capsize=2, color=C_ST, edgecolor="k",
               linewidth=0.3, error_kw={"linewidth": 0.6}, label="$S_T$")
        ax.axhline(0, color="gray", linewidth=0.5)
        if r == 0:
            ax.set_title(fs.PLABEL[p], fontsize=12)
        if c == 0:
            ax.set_ylabel(METRIC_LABEL[metric], fontsize=8, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([fs.label(s) for s in STATIONS], rotation=60, fontsize=6)   # black (default)
        ax.grid(True, axis="y", alpha=0.25)
        ax.label_outer()
axes[0][0].set_ylim(-0.05, 1.05)
# legend in the (row 1, col 3) panel, which is empty since recruitment does not drive the magnitude
axes[0][2].legend(loc="center", fontsize=8, frameon=True)
fig.tight_layout()
fs.save(fig, "Figure_6", subdir=None)
