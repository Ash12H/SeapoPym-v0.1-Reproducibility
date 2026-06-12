"""Figure 3 — theoretical benchmark: convergence to the analytical asymptote B = R / lambda.

Numerical convergence of the SeapoPym biomass to the steady state B = R / lambda(T) (recruitment
R = E * NPP) under constant forcing, at four temperatures (0/10/20/30 C), NPP = 300 mg C m-2 d-1.
Line COLOUR = temperature; DASHED line = analytical asymptote; x axis = elapsed days (log).

Reads ONLY the committed product products/benchmark_convergence.csv (produced by
scripts/experiments/run_benchmark.py) — no framework needed at figure time.
Output : figures/Figure_3.{pdf,png}
Run    : .venv/bin/python scripts/figures/fig03_benchmark.py
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

from seapopym_repro import figstyle as fs, paths

df = pd.read_csv(paths.PRODUCTS / "benchmark_convergence.csv")
temperatures = sorted(df["temperature"].unique())
DURATION_DAYS = int(df["day"].max())

fig, ax = plt.subplots(figsize=(0.70 * fs.WIDTH_FULL, 0.50 * fs.WIDTH_FULL))
colors = [fs.temp_color(t) for t in temperatures]   # colour = temperature, on the station thermal palette
for t, color in zip(temperatures, colors, strict=True):
    d = df[df["temperature"] == t]
    ax.plot(d["day"], d["biomass"], lw=1.6, color=color)
    ax.axhline(d["asymptote"].iloc[0], ls="--", lw=1.4, color=color, alpha=0.85)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(1, DURATION_DAYS)
ax.set_ylim(50, 10000)
plain = FuncFormatter(lambda value, _: f"{value:g}")  # 1, 10, 100, 1000 instead of 10^n
ax.xaxis.set_major_formatter(plain)
ax.yaxis.set_major_formatter(plain)
ax.set_xlabel("Elapsed time (days)")
ax.set_ylabel(r"Biomass (mg m$^{-2}$)")
ax.grid(True, which="both", alpha=0.3, linewidth=0.5)

# Legend: colour = temperature, dashed = analytical asymptote B = R / lambda.
colour_handles = [Line2D([0], [0], color=c, lw=2, label=f"{t:g} °C")
                  for t, c in zip(temperatures, colors, strict=True)]
asymptote_handle = Line2D([0], [0], color="black", lw=1.4, ls="--", label=r"$B = R/\lambda$")
ax.legend(handles=[*colour_handles, asymptote_handle], loc="upper left", ncol=2, frameon=True)

fig.tight_layout()
fs.save(fig, "Figure_3", subdir=None)
