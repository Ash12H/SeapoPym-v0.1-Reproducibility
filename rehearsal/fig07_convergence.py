"""fig6_convergence.py — REHEARSAL Figure 6 (GA convergence).

Best-so-far NRMSE vs generation, one curve per experiment, from the logbooks in
rehearsal/logbooks/. The cost is read from the aggregate Weighted_fitness
(= -weighted cost), so it works for single-station (1 obs) and MERGED (6 obs) alike.
A marker shows the early-stopping generation (= last generation in the logbook).

Doubles as the diagnostic for the patience/tol choice: if the curve is flat well
before the stop generation, the run spent generations chasing <tol improvements.

Output: rehearsal/figures/fig6_convergence.png
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
LOGS = HERE / "logbooks"
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)
WF = ("Weighted_fitness", "Weighted_fitness")
ORDER = ["BARENTS", "PAPA", "Bay_of_Biscay", "BATS", "Canaries", "HOT", "MERGED"]


def best_so_far(path):
    df = pd.read_parquet(path)
    # per-generation best aggregate cost = -max(Weighted_fitness); then cumulative min
    per_gen = -df[WF].groupby(level="Generation").max()
    return per_gen.index.values, np.minimum.accumulate(per_gen.values)


logs = {p.stem.replace("ga_logbook_", ""): p for p in sorted(LOGS.glob("ga_logbook_*.parquet"))}
present = [e for e in ORDER if e in logs] + [e for e in logs if e not in ORDER]

fig, ax = plt.subplots(figsize=(9, 5.2), dpi=200)
colors = dict(zip(present, plt.cm.tab10(np.linspace(0, 1, max(len(present), 1)))))
for exp in present:
    gens, cost = best_so_far(logs[exp])
    ax.plot(gens, cost, lw=1.6, color=colors[exp], label=f"{exp} (stop @ {int(gens.max())})")
    ax.scatter([gens.max()], [cost[-1]], color=colors[exp], s=40, zorder=5, edgecolor="black", lw=0.5)

ax.set_xlabel("Generation")
ax.set_ylabel("Best-so-far cost (NRMSE)")
ax.set_title("GA convergence — best-so-far cost per generation\n(SBX, pop 256, early stopping)",
             fontweight="bold", fontsize=11)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9)
plt.tight_layout()
out = FIG / "Figure_7.png"
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
print("wrote", out)

# diagnostic: at which gen does best-so-far reach within 1% / 2% of its final value?
for exp in present:
    gens, cost = best_so_far(logs[exp])
    final = cost[-1]
    for thr in (0.02, 0.01):
        reached = gens[np.where(cost <= final * (1 + thr))[0][0]]
        print(f"{exp:14s} within {thr*100:.0f}% of final ({final:.4g}) by gen {int(reached)}  (stopped @ {int(gens.max())})")
