"""Simplified Figure 7 — best-individual parameters across the 7 twin experiments.

Twin experiments (synthetic observations): 6 single stations + MERGED.
For each optimisation we take the best individual (lowest NRMSE) and show its
5 parameters in two beeswarm variants:

  A) recovery error  : MAPE = |optimised - reference| / |reference| * 100  (one panel, log y)
  B) raw value       : the parameter value within the optimisation bounds, with the
                       reference value marked                              (5 panels)

Points are placed as a true beeswarm: each point hugs the central vertical axis
and is nudged sideways only as far as needed to avoid overlapping its neighbours.
Colour identifies the optimisation; MERGED is drawn as a black star.

NOTE on bounds: the existing twin logbooks were optimised with the ORIGINAL
bounds, so this figure uses those (a parameter "value within the bounds" must use
the bounds that were actually used). They are hard-coded below.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data"
FIG = HERE / "figures"

EXPERIMENTS = ["BARENTS", "PAPA", "Bay_of_Biscay", "BATS", "Canaries", "HOT", "MERGED"]
DISPLAY = {"BARENTS": "BARENTS", "PAPA": "PAPA", "Bay_of_Biscay": "BISCAY",
           "BATS": "BATS", "Canaries": "CANARY", "HOT": "HOT", "MERGED": "MERGED"}
PARAM_ORDER = ["energy_transfert", "tr_0", "gamma_tr", "lambda_temperature_0", "gamma_lambda_temperature"]
LABELS = {"energy_transfert": r"$E$", "tr_0": r"$\tau_{r_0}$", "gamma_tr": r"$\gamma_{\tau_r}$",
          "lambda_temperature_0": r"$\lambda_0$", "gamma_lambda_temperature": r"$\gamma_\lambda$"}

REF = yaml.safe_load(open(ROOT / "parameters.yaml"))["model_parameters"]["reference"]
BOUNDS = {  # original bounds used by the existing twin run
    "energy_transfert": [0.0, 0.5], "tr_0": [0.0, 100.0], "gamma_tr": [-0.5, 0.0],
    "lambda_temperature_0": [0.002, 0.25], "gamma_lambda_temperature": [0.0, 0.5],
}

best = {}
for sid in EXPERIMENTS:
    df = pd.read_parquet(DATA / f"ga_logbook_{sid}_validation.parquet")
    best[sid] = df.loc[df[("Weighted_fitness", "Weighted_fitness")].idxmax(), "Parametre"].to_dict()

colors = dict(zip(EXPERIMENTS, plt.cm.tab10(np.linspace(0, 1, len(EXPERIMENTS)))))
colors["MERGED"] = (0.1, 0.1, 0.1, 1.0)


def _style(sid):
    if sid == "MERGED":
        return dict(marker="*", s=340, color=colors[sid], edgecolor="white", linewidth=0.9, zorder=6)
    return dict(marker="o", s=95, color=colors[sid], edgecolor="black", linewidth=0.5, zorder=5)


def beeswarm(yvals, ythr, xstep, max_lane=8):
    """Centred beeswarm x-offsets: points hug x=0, nudged sideways only to avoid overlap."""
    yvals = np.asarray(yvals, dtype=float)
    lanes = [0.0]
    for m in range(1, max_lane + 1):
        lanes += [m * xstep, -m * xstep]
    placed, offs = [], np.zeros(len(yvals))
    for idx in np.argsort(yvals):  # place from low to high y
        yi = yvals[idx]
        for c in lanes:
            if all(not (abs(yi - py) < ythr and abs(c - px) < xstep * 0.99) for py, px in placed):
                offs[idx] = c
                placed.append((yi, c))
                break
    return offs


# ---------------- Variant A: recovery MAPE (log y) ----------------
fig, ax = plt.subplots(figsize=(9, 5.2), dpi=200)
for j, p in enumerate(PARAM_ORDER):
    mapes = np.array([abs(best[sid][p] - REF[p]) / abs(REF[p]) * 100 for sid in EXPERIMENTS])
    offs = beeswarm(np.log10(mapes), ythr=0.085, xstep=0.05)
    for off, sid, m in zip(offs, EXPERIMENTS, mapes):
        ax.scatter(j + off, m, **_style(sid))
ax.axhline(10, color="gray", linestyle="--", linewidth=1, alpha=0.7)
ax.text(len(PARAM_ORDER) - 0.55, 10.6, "10 %", color="gray", fontsize=9, va="bottom", ha="right")
ax.set_yscale("log")
ax.set_xticks(range(len(PARAM_ORDER)))
ax.set_xticklabels([LABELS[p] for p in PARAM_ORDER], fontsize=13)
ax.set_xlim(-0.5, len(PARAM_ORDER) - 0.5)
ax.set_ylabel("Recovery error  MAPE (%)")
ax.set_title("Twin experiments — parameter recovery of the best individual", fontweight="bold", fontsize=11)
ax.grid(True, axis="y", alpha=0.3)
handles = [Line2D([0], [0], marker="*" if s == "MERGED" else "o", color="w",
                  markerfacecolor=colors[s], markeredgecolor="black",
                  markersize=15 if s == "MERGED" else 9, linestyle="", label=DISPLAY[s]) for s in EXPERIMENTS]
fig.legend(handles=handles, loc="lower center", ncol=7, frameon=False, bbox_to_anchor=(0.5, -0.02), fontsize=9)
plt.tight_layout(rect=[0, 0.05, 1, 1])
fig.savefig(FIG / "Figure_7A_recovery_mape.png", dpi=200, bbox_inches="tight", facecolor="white")
print("wrote", FIG / "Figure_7A_recovery_mape.png")

# ---------------- Variant B: raw value within bounds ----------------
fig, axes = plt.subplots(1, 5, figsize=(13, 4.6), dpi=200)
for ax, p in zip(axes, PARAM_ORDER):
    lo, hi = BOUNDS[p]
    vals = np.array([best[sid][p] for sid in EXPERIMENTS])
    offs = beeswarm(vals, ythr=(hi - lo) * 0.05, xstep=0.075)
    for off, sid, v in zip(offs, EXPERIMENTS, vals):
        ax.scatter(off, v, **_style(sid))
    ax.axhline(REF[p], color="gray", linestyle="--", linewidth=1.3)
    ax.set_ylim(lo, hi)
    ax.set_xlim(-0.6, 0.6)
    ax.set_xticks([])
    ax.set_title(LABELS[p], fontsize=13)
    ax.grid(True, axis="y", alpha=0.3)
axes[0].set_ylabel("Parameter value (within bounds)")
handles2 = handles + [Line2D([0], [0], color="gray", linestyle="--", label="Reference value")]
fig.legend(handles=handles2, loc="lower center", ncol=8, frameon=False, bbox_to_anchor=(0.5, -0.05), fontsize=9)
fig.suptitle("Twin experiments — best-individual parameter values across the 7 optimisations",
             fontweight="bold", fontsize=11)
plt.tight_layout(rect=[0, 0.04, 1, 0.96])
fig.savefig(FIG / "Figure_7B_values.png", dpi=200, bbox_inches="tight", facecolor="white")
print("wrote", FIG / "Figure_7B_values.png")
