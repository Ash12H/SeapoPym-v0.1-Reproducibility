"""rehearsal/cmaes_figures.py — figures from the seeded CMA-ES ensemble (rehearsal/cmaes/).

Reads the production results written by run_cmaes_seed_ensemble.py:
    rehearsal/cmaes/cmaes_seed_ensemble_l{L}.csv              per (exp, seed): best_nrmse + 5 params
    rehearsal/cmaes/seeds/ga_logbook_{exp}_lambda{L}_seed{S}.parquet   per-seed trajectory

Produces (rehearsal/figures/cmaes/):
    cmaes_convergence.png        best-so-far NRMSE vs evaluations, all seeds, one panel per experiment
    cmaes_recovery_values.png    recovered parameter values per experiment (per-seed strip + best + reference)
    cmaes_recovery_error.png     MAPE = |best - ref|/|ref|*100 of the best individual (beeswarm, log y)

Compare on PARAMETER RECOVERY vs reference, not on NRMSE (low NRMSE on warm stations is equifinality).
Run: .venv/bin/python rehearsal/cmaes_figures.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
CMA = ROOT / "rehearsal" / "cmaes"
FIG = ROOT / "rehearsal" / "figures" / "cmaes"
FIG.mkdir(parents=True, exist_ok=True)
WF = ("Weighted_fitness", "Weighted_fitness")

# MERGED leftmost, then single stations coldest -> warmest (mean SST: BARENTS 1, PAPA 9, BISCAY 14,
# CANARY 19, BATS 21, HOT 24 °C) — same cold->warm convention as the Sobol figure.
ORDER = ["MERGED", "BARENTS", "PAPA", "Bay_of_Biscay", "Canaries", "BATS", "HOT"]
DISPLAY = {"BARENTS": "BARENTS", "PAPA": "PAPA", "Bay_of_Biscay": "BISCAY", "BATS": "BATS",
           "Canaries": "CANARY", "HOT": "HOT", "MERGED": "MERGED"}
PARAM_ORDER = ["energy_transfert", "tr_0", "gamma_tr", "lambda_temperature_0", "gamma_lambda_temperature"]
PLABEL = {"energy_transfert": r"$E$", "tr_0": r"$\tau_{r_0}$", "gamma_tr": r"$\gamma_{\tau_r}$",
          "lambda_temperature_0": r"$\lambda_0$", "gamma_lambda_temperature": r"$\gamma_\lambda$"}

_p = yaml.safe_load(open(ROOT / "parameters.yaml"))["model_parameters"]
REF, BOUNDS = _p["reference"], _p["bounds"]

csvs = sorted(CMA.glob("cmaes_seed_ensemble_l*.csv"))
if not csvs:
    raise SystemExit(f"No CMA-ES results CSV in {CMA} — run run_cmaes_seed_ensemble.py first")
d = pd.read_csv(csvs[-1])
LAM = int(d["lambda"].iloc[0])
EXP = [e for e in ORDER if e in set(d.experiment)]
NSEEDS = int(d.groupby("experiment").seed.nunique().max())
print(f"{csvs[-1].name}: λ={LAM}, {len(EXP)} experiments, up to {NSEEDS} seeds")

colors = dict(zip(EXP, plt.cm.tab10(np.linspace(0, 1, max(len(EXP), 1)))))
if "MERGED" in colors:
    colors["MERGED"] = (0.1, 0.1, 0.1, 1.0)


def best_so_far(path):
    df = pd.read_parquet(path)
    per_gen = df[WF].groupby(level="Generation").max().sort_index()
    return (per_gen.index.to_numpy() + 1) * LAM, -per_gen.cummax().to_numpy()


def _style(e):
    if e == "MERGED":
        return dict(marker="*", s=320, color=colors[e], edgecolor="white", linewidth=0.9, zorder=6)
    return dict(marker="o", s=90, color=colors[e], edgecolor="black", linewidth=0.5, zorder=5)


def beeswarm(yvals, ythr, xstep, max_lane=8):
    yvals = np.asarray(yvals, float)
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


# ---------------- Figure 1: convergence (all seeds, one panel per experiment) ----------------
def fig_convergence():
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), squeeze=False)
    axes = axes.ravel()
    for k, exp in enumerate(EXP):
        ax = axes[k]
        for p in sorted((CMA / "seeds").glob(f"ga_logbook_{exp}_lambda{LAM}_seed*.parquet")):
            ev, b = best_so_far(p)
            ax.plot(ev, b, lw=0.9, alpha=0.6, color=colors[exp])
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(DISPLAY.get(exp, exp), fontweight="bold", fontsize=11)
        ax.set_xlabel("model evaluations", fontsize=8); ax.set_ylabel("best-so-far NRMSE", fontsize=8)
        ax.grid(True, which="both", alpha=0.25)
    for j in range(len(EXP), len(axes)):
        axes[j].axis("off")
    fig.suptitle(f"CMA-ES convergence — all seeds (λ={LAM}, up to {NSEEDS} restarts)", fontweight="bold", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIG / "cmaes_convergence.png", dpi=200, bbox_inches="tight", facecolor="white")
    print("wrote", FIG / "cmaes_convergence.png")
    plt.close(fig)


# ---------------- Figure 2: recovered parameter values (per-seed strip + best + reference) ----------------
def fig_recovery_values():
    best_row = d.loc[d.groupby("experiment").best_nrmse.idxmin().values].set_index("experiment")
    fig, axes = plt.subplots(1, 5, figsize=(15, 4.6), dpi=200)
    x = np.arange(len(EXP))
    for ax, p in zip(axes, PARAM_ORDER):
        # y-axis centred on the reference; half-range = 1.2 * largest |recovered - reference|
        D = 1.2 * float(np.max(np.abs(d[p].to_numpy() - REF[p]))) or 1.0
        for j, e in enumerate(EXP):
            vals = d.loc[d.experiment == e, p].to_numpy()
            jit = (np.random.RandomState(0).rand(len(vals)) - 0.5) * 0.5
            ax.scatter(np.full_like(vals, x[j]) + jit, vals, s=14, alpha=0.4, color=colors[e], zorder=3)
            ax.scatter(x[j], best_row.loc[e, p], **_style(e))   # best individual highlighted
        ax.axhline(REF[p], color="gray", linestyle="--", linewidth=1.3, zorder=1)
        ax.set_ylim(REF[p] - D, REF[p] + D); ax.set_xticks(x)
        ax.set_xticklabels([DISPLAY[e] for e in EXP], rotation=60, fontsize=7)
        ax.set_title(PLABEL[p], fontsize=13); ax.grid(True, axis="y", alpha=0.3)
    axes[0].set_ylabel("Parameter value (within bounds)")
    handles = [Line2D([0], [0], color="gray", ls="--", label="reference"),
               Line2D([0], [0], marker="o", color="w", markerfacecolor="0.5", markeredgecolor="k", label="best individual")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.04), fontsize=9)
    fig.suptitle(f"Recovered parameters — per-seed spread + best (CMA-ES, λ={LAM}, {NSEEDS} restarts)",
                 fontweight="bold", fontsize=12)
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    fig.savefig(FIG / "cmaes_recovery_values.png", dpi=200, bbox_inches="tight", facecolor="white")
    print("wrote", FIG / "cmaes_recovery_values.png")
    plt.close(fig)


# ---------------- Figure 3: recovery error (MAPE) of the best individual ----------------
def fig_recovery_error():
    best = {e: d.loc[d[d.experiment == e].best_nrmse.idxmin()] for e in EXP}
    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=200)
    for j, p in enumerate(PARAM_ORDER):
        mapes = np.array([abs(best[e][p] - REF[p]) / abs(REF[p]) * 100 for e in EXP])
        offs = beeswarm(np.log10(np.clip(mapes, 1e-2, None)), ythr=0.085, xstep=0.05)
        for off, e, m in zip(offs, EXP, mapes):
            ax.scatter(j + off, max(m, 1e-2), **_style(e))
    ax.axhline(10, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.text(len(PARAM_ORDER) - 0.55, 10.6, "10 %", color="gray", fontsize=9, va="bottom", ha="right")
    ax.set_yscale("log"); ax.set_xticks(range(len(PARAM_ORDER)))
    ax.set_xticklabels([PLABEL[p] for p in PARAM_ORDER], fontsize=13)
    ax.set_xlim(-0.5, len(PARAM_ORDER) - 0.5)
    ax.set_ylabel("Recovery error  MAPE (%)")
    ax.set_title(f"Parameter recovery of the best individual (CMA-ES, λ={LAM})", fontweight="bold", fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)
    handles = [Line2D([0], [0], marker="*" if e == "MERGED" else "o", color="w", markerfacecolor=colors[e],
                      markeredgecolor="black", markersize=15 if e == "MERGED" else 9, linestyle="",
                      label=DISPLAY.get(e, e)) for e in EXP]
    fig.legend(handles=handles, loc="lower center", ncol=min(len(EXP), 7), frameon=False,
               bbox_to_anchor=(0.5, -0.02), fontsize=9)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(FIG / "cmaes_recovery_error.png", dpi=200, bbox_inches="tight", facecolor="white")
    print("wrote", FIG / "cmaes_recovery_error.png")
    plt.close(fig)


if __name__ == "__main__":
    fig_convergence()
    fig_recovery_values()
    fig_recovery_error()
