"""HOT vs BATS — optimised parameters against real observations.

One panel per parameter. A dashed horizontal line marks the original (reference)
value. Two floating bars (BATS in blue, HOT in red) run from the reference line
to the optimised value: the bar length is the deviation from the original, its
direction (up/down) the sign.

Uses the real-data optimisation logbooks (widened bounds), and the bounds in
parameters.yaml as the per-panel y-axis (those are the bounds these runs used).
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data"
FIG = HERE / "figures"

PARAM_ORDER = ["energy_transfert", "tr_0", "gamma_tr", "lambda_temperature_0", "gamma_lambda_temperature"]
LABELS = {"energy_transfert": r"$E$", "tr_0": r"$\tau_{r_0}$ (day)", "gamma_tr": r"$\gamma_{\tau_r}$ ($\degree$C$^{-1}$)",
          "lambda_temperature_0": r"$\lambda_0$ (day$^{-1}$)", "gamma_lambda_temperature": r"$\gamma_\lambda$ ($\degree$C$^{-1}$)"}

P = yaml.safe_load(open(ROOT / "parameters.yaml"))["model_parameters"]
REF, BOUNDS = P["reference"], P["bounds"]

COLORS = {"BATS": "#1f77b4", "HOT": "#d62728"}  # blue / red
LOGBOOK = {"HOT": "ga_logbook_HOT_validation_hotreal.parquet",
           "BATS": "ga_logbook_BATS_validation_batsreal.parquet"}

opt = {}
for s, f in LOGBOOK.items():
    df = pd.read_parquet(DATA / f)
    opt[s] = df.loc[df[("Weighted_fitness", "Weighted_fitness")].idxmax(), "Parametre"].to_dict()

order = ["BATS", "HOT"]  # blue, red
xpos = {"BATS": 0.0, "HOT": 1.0}
width = 0.62

fig, axes = plt.subplots(1, 5, figsize=(13, 4.6), dpi=200)
for ax, p in zip(axes, PARAM_ORDER):
    ref = REF[p]
    for s in order:
        v = opt[s][p]
        ax.bar(xpos[s], height=v - ref, bottom=ref, width=width, color=COLORS[s],
               edgecolor="black", linewidth=0.6, zorder=3)
        # annotate the optimised value just inside the free end of the bar (never overflows the panel)
        up = v >= ref
        ax.annotate(f"{v:.3g}", (xpos[s], v), ha="center",
                    va="top" if up else "bottom",
                    xytext=(0, -4 if up else 4), textcoords="offset points",
                    fontsize=8.5, zorder=5,
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.75))
    ax.axhline(ref, color="gray", linestyle="--", linewidth=1.4, zorder=2)
    ax.set_ylim(*BOUNDS[p])
    ax.set_xlim(-0.7, 1.7)
    ax.set_xticks([])
    ax.set_title(LABELS[p], fontsize=12)
    ax.grid(True, axis="y", alpha=0.3)

axes[0].set_ylabel("Parameter value")
legend = [
    Patch(facecolor=COLORS["BATS"], edgecolor="black", label="BATS"),
    Patch(facecolor=COLORS["HOT"], edgecolor="black", label="HOT"),
    Line2D([0], [0], color="gray", linestyle="--", label="Original (reference) value"),
]
fig.legend(handles=legend, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.04), fontsize=10)
fig.suptitle("Optimised parameters against real observations — HOT vs BATS",
             fontweight="bold", fontsize=12)
plt.tight_layout(rect=[0, 0.04, 1, 0.95])
out = FIG / "HOTBATS_param_deviation_bars.png"
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
print("wrote", out)
print("HOT :", {k: round(v, 4) for k, v in opt["HOT"].items()})
print("BATS:", {k: round(v, 4) for k, v in opt["BATS"].items()})
