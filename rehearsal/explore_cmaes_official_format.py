"""rehearsal/explore_cmaes_official_format.py — CMA-ES results in the OFFICIAL figure format.

Re-renders the seeded CMA-ES cross-check (best seed of 10 per experiment) using the EXACT layout
of the official Figure_7 / Figure_8 / Figure_8_recovery_error, so the improvement can be compared
side by side with the GA figures. Reads the best-seed CMA-ES logbooks in logbooks/cmaes/ (GA-format,
written by run_cmaes_seed_ensemble.py). EXPLORATORY: writes to figures/exploratory/ with a _cmaes
suffix; never touches the official Figure_*.png.

  Figure_7_cmaes.png                : best-so-far cost (NRMSE) vs generation, one curve per experiment
  Figure_8_recovery_error_cmaes.png : recovery MAPE = |opt-ref|/|ref|*100 (beeswarm, log y)
  Figure_8_cmaes.png                : best-individual parameter value within the current bounds

Run: .venv/bin/python rehearsal/explore_cmaes_official_format.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
LOGS = HERE / "logbooks" / "cmaes"          # best-seed CMA-ES logbooks (GA-compatible)
FIG = HERE / "figures" / "exploratory"
FIG.mkdir(parents=True, exist_ok=True)
WF = ("Weighted_fitness", "Weighted_fitness")
LAMBDA = int(4 + 3 * np.log(5))             # DEAP default for the 5 params (= 8)
SUBTITLE = f"CMA-ES, best of 10 restarts (λ={LAMBDA})"

ORDER = ["BARENTS", "PAPA", "Bay_of_Biscay", "BATS", "Canaries", "HOT", "MERGED"]
DISPLAY = {"BARENTS": "BARENTS", "PAPA": "PAPA", "Bay_of_Biscay": "BISCAY",
           "BATS": "BATS", "Canaries": "CANARY", "HOT": "HOT", "MERGED": "MERGED"}
PARAM_ORDER = ["energy_transfert", "tr_0", "gamma_tr", "lambda_temperature_0", "gamma_lambda_temperature"]
LABELS = {"energy_transfert": r"$E$", "tr_0": r"$\tau_{r_0}$", "gamma_tr": r"$\gamma_{\tau_r}$",
          "lambda_temperature_0": r"$\lambda_0$", "gamma_lambda_temperature": r"$\gamma_\lambda$"}

_p = yaml.safe_load(open(ROOT / "parameters.yaml"))["model_parameters"]
REF = _p["reference"]
BOUNDS = _p["bounds"]

logs = {p.stem.replace("ga_logbook_", ""): p for p in sorted(LOGS.glob("ga_logbook_*.parquet"))}
EXP = [e for e in ORDER if e in logs] + [e for e in logs if e not in ORDER]
if not EXP:
    raise SystemExit(f"No CMA-ES logbooks found in {LOGS} — run run_cmaes_seed_ensemble.py first")
print("experiments present:", EXP)

best = {e: pd.read_parquet(logs[e]).loc[lambda d: d[WF].idxmax(), "Parametre"].to_dict() for e in EXP}
colors = dict(zip(EXP, plt.cm.tab10(np.linspace(0, 1, max(len(EXP), 1)))))
if "MERGED" in colors:
    colors["MERGED"] = (0.1, 0.1, 0.1, 1.0)


def _style(e):
    if e == "MERGED":
        return dict(marker="*", s=340, color=colors[e], edgecolor="white", linewidth=0.9, zorder=6)
    return dict(marker="o", s=95, color=colors[e], edgecolor="black", linewidth=0.5, zorder=5)


def beeswarm(yvals, ythr, xstep, max_lane=8):
    yvals = np.asarray(yvals, dtype=float)
    lanes = [0.0]
    for m in range(1, max_lane + 1):
        lanes += [m * xstep, -m * xstep]
    placed, offs = [], np.zeros(len(yvals))
    for idx in np.argsort(yvals):
        yi = yvals[idx]
        for c in lanes:
            if all(not (abs(yi - py) < ythr and abs(c - px) < xstep * 0.99) for py, px in placed):
                offs[idx] = c
                placed.append((yi, c))
                break
    return offs


# ---------------- Figure 7 (CMA-ES): convergence ----------------
def best_so_far(path):
    df = pd.read_parquet(path)
    per_gen = -df[WF].groupby(level="Generation").max()
    return per_gen.index.values, np.minimum.accumulate(per_gen.values)


fig, ax = plt.subplots(figsize=(9, 5.2), dpi=200)
conv_colors = dict(zip(EXP, plt.cm.tab10(np.linspace(0, 1, max(len(EXP), 1)))))
for exp in EXP:
    gens, cost = best_so_far(logs[exp])
    ax.plot(gens, cost, lw=1.6, color=conv_colors[exp], label=f"{DISPLAY.get(exp, exp)} (stop @ {int(gens.max())})")
    ax.scatter([gens.max()], [cost[-1]], color=conv_colors[exp], s=40, zorder=5, edgecolor="black", lw=0.5)
ax.set_xlabel("Generation")
ax.set_ylabel("Best-so-far cost (NRMSE)")
ax.set_title(f"CMA-ES convergence — best-so-far cost per generation\n({SUBTITLE})",
             fontweight="bold", fontsize=11)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9)
plt.tight_layout()
fig.savefig(FIG / "Figure_7_cmaes.png", dpi=200, bbox_inches="tight", facecolor="white")
print("wrote", FIG / "Figure_7_cmaes.png")
plt.close(fig)

# ---------------- Figure 8 recovery error (CMA-ES): MAPE (log y) ----------------
fig, ax = plt.subplots(figsize=(9, 5.2), dpi=200)
for j, p in enumerate(PARAM_ORDER):
    mapes = np.array([abs(best[e][p] - REF[p]) / abs(REF[p]) * 100 for e in EXP])
    offs = beeswarm(np.log10(np.clip(mapes, 1e-2, None)), ythr=0.085, xstep=0.05)
    for off, e, m in zip(offs, EXP, mapes):
        ax.scatter(j + off, max(m, 1e-2), **_style(e))
ax.axhline(10, color="gray", linestyle="--", linewidth=1, alpha=0.7)
ax.text(len(PARAM_ORDER) - 0.55, 10.6, "10 %", color="gray", fontsize=9, va="bottom", ha="right")
ax.set_yscale("log")
ax.set_xticks(range(len(PARAM_ORDER)))
ax.set_xticklabels([LABELS[p] for p in PARAM_ORDER], fontsize=13)
ax.set_xlim(-0.5, len(PARAM_ORDER) - 0.5)
ax.set_ylabel("Recovery error  MAPE (%)")
ax.set_title(f"Twin experiments — parameter recovery of the best individual ({SUBTITLE})",
             fontweight="bold", fontsize=11)
ax.grid(True, axis="y", alpha=0.3)
handles = [Line2D([0], [0], marker="*" if e == "MERGED" else "o", color="w", markerfacecolor=colors[e],
                  markeredgecolor="black", markersize=15 if e == "MERGED" else 9, linestyle="",
                  label=DISPLAY.get(e, e)) for e in EXP]
fig.legend(handles=handles, loc="lower center", ncol=min(len(EXP), 7), frameon=False,
           bbox_to_anchor=(0.5, -0.02), fontsize=9)
plt.tight_layout(rect=[0, 0.05, 1, 1])
fig.savefig(FIG / "Figure_8_recovery_error_cmaes.png", dpi=200, bbox_inches="tight", facecolor="white")
print("wrote", FIG / "Figure_8_recovery_error_cmaes.png")
plt.close(fig)

# ---------------- Figure 8 (CMA-ES): value within bounds ----------------
fig, axes = plt.subplots(1, 5, figsize=(13, 4.6), dpi=200)
for ax, p in zip(axes, PARAM_ORDER):
    lo, hi = BOUNDS[p]
    vals = np.array([best[e][p] for e in EXP])
    offs = beeswarm(vals, ythr=(hi - lo) * 0.05, xstep=0.075)
    for off, e, v in zip(offs, EXP, vals):
        ax.scatter(off, v, **_style(e))
    ax.axhline(REF[p], color="gray", linestyle="--", linewidth=1.3)
    ax.set_ylim(lo, hi)
    ax.set_xlim(-0.6, 0.6)
    ax.set_xticks([])
    ax.set_title(LABELS[p], fontsize=13)
    ax.grid(True, axis="y", alpha=0.3)
axes[0].set_ylabel("Parameter value (within bounds)")
handles2 = handles + [Line2D([0], [0], color="gray", linestyle="--", label="Reference value")]
fig.legend(handles=handles2, loc="lower center", ncol=min(len(EXP) + 1, 8), frameon=False,
           bbox_to_anchor=(0.5, -0.05), fontsize=9)
fig.suptitle(f"Twin experiments — best-individual parameter values ({SUBTITLE})", fontweight="bold", fontsize=11)
plt.tight_layout(rect=[0, 0.04, 1, 0.96])
fig.savefig(FIG / "Figure_8_cmaes.png", dpi=200, bbox_inches="tight", facecolor="white")
print("wrote", FIG / "Figure_8_cmaes.png")
plt.close(fig)
