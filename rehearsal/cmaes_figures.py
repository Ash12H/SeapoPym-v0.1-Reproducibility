"""rehearsal/cmaes_figures.py — figures from the seeded CMA-ES ensemble.

Reads ONLY the committed frozen products written by run_cmaes_seed_ensemble.py (the display
contract — never a raw logbook, never a recomputation):
    cmaes/cmaes_seed_ensemble_l{L}.csv          per (exp, seed): best_nrmse + 5 recovered params
    cmaes/cmaes_convergence_traces_l{L}.csv      per (exp, seed): best-so-far NRMSE vs evaluations

Per-station colour + marker, typography, GMD widths and PDF+PNG saving all come from figstyle.py
(single source of truth). Produces (rehearsal/figures/cmaes/, each as .pdf + .png):
    cmaes_convergence       best-so-far NRMSE vs evaluations, all seeds, one panel per experiment
    cmaes_recovery_values   recovered parameter values per experiment (per-seed strip + best + ref)
    cmaes_recovery_error    MAPE = |best - ref| / |ref| * 100 of the best individual (beeswarm, log y)

Compare on PARAMETER RECOVERY vs reference, not on NRMSE (low NRMSE on warm stations is equifinality).
Run: .venv/bin/python rehearsal/cmaes_figures.py
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from seapopym_repro import figstyle as fs, paths

CMA = paths.PRODUCTS   # the committed frozen CSV products (recovery + convergence traces)
REF, PARAM_ORDER, PLABEL = fs.REF, fs.PARAM_ORDER, fs.PLABEL


def _load_csv(pattern):
    hits = sorted(CMA.glob(pattern))
    if not hits:
        raise SystemExit(f"No {pattern} in {CMA} — run run_cmaes_seed_ensemble.py first")
    return pd.read_csv(hits[-1])


d = _load_csv("cmaes_seed_ensemble_l*.csv")                  # recovery values
traces = _load_csv("cmaes_convergence_traces_l*.csv")        # convergence trajectories
LAM = int(d["lambda"].iloc[0])
EXP = fs.order(set(d.experiment))                            # canonical order, only what's present
NSEEDS = int(d.groupby("experiment").seed.nunique().max())
print(f"λ={LAM}, {len(EXP)} experiments, up to {NSEEDS} seeds")


def beeswarm(yvals, ythr, xstep, max_lane=8):
    """Spread overlapping points into discrete x-lanes so none collide (log-space y expected)."""
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
    fig, axes = plt.subplots(2, 4, figsize=(fs.WIDTH_FULL, 0.62 * fs.WIDTH_FULL),
                             sharex=True, sharey=True, squeeze=False)
    axes = axes.ravel()
    for k, exp in enumerate(EXP):
        ax = axes[k]
        for _, t in traces[traces.experiment == exp].groupby("seed"):
            t = t.sort_values("evaluations")
            ax.plot(t.evaluations, t.best_nrmse, lw=0.9, alpha=0.55, color=fs.color(exp),
                    solid_capstyle="round")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(fs.label(exp), color=fs.color(exp), pad=3)
        ax.grid(True, which="both", alpha=0.22, linewidth=0.5)
    for j in range(len(EXP), len(axes)):
        axes[j].axis("off")
    if len(EXP) < len(axes):                 # top-row panel above an empty cell keeps its own x ticks
        axes[len(EXP) - 1].tick_params(labelbottom=True)
    fig.supxlabel("model evaluations", fontsize=9)
    fig.supylabel("best-so-far NRMSE", fontsize=9)
    fig.tight_layout()
    fs.save(fig, "cmaes_convergence")
    plt.close(fig)


# ---------------- Figure 2: recovered parameter values (per-seed strip + best + reference) ----------------
def fig_recovery_values():
    best_row = d.loc[d.groupby("experiment").best_nrmse.idxmin().values].set_index("experiment")
    fig, axes = plt.subplots(1, 5, figsize=(fs.WIDTH_FULL, 0.34 * fs.WIDTH_FULL))
    x = np.arange(len(EXP))
    jit = (np.random.RandomState(0).rand(NSEEDS) - 0.5) * 0.55
    for ax, p in zip(axes, PARAM_ORDER):
        half = 1.2 * float(np.max(np.abs(d[p].to_numpy() - REF[p]))) or 1.0   # y centred on reference
        ax.axhline(REF[p], color="0.55", linestyle="--", linewidth=1.1, zorder=1)
        for j, e in enumerate(EXP):
            vals = d.loc[d.experiment == e, p].to_numpy()
            ax.scatter(x[j] + jit[:len(vals)], vals, s=12, alpha=0.40, color=fs.color(e),
                       edgecolor="none", zorder=3)                            # per-seed restarts
            fs.scatter(ax, x[j], best_row.loc[e, p], e, base=46)              # best individual
        ax.set_ylim(REF[p] - half, REF[p] + half)
        ax.set_xlim(-0.6, len(EXP) - 0.4)
        ax.set_xticks([])                          # station identity = colour + marker + the shared legend
        ax.set_title(PLABEL[p], fontsize=12); ax.grid(True, axis="y", alpha=0.3)
    axes[0].set_ylabel("recovered value")
    handles = fs.legend_handles(EXP) + [Line2D([0], [0], color="0.55", ls="--", label="reference")]
    fig.legend(handles=handles, loc="lower center", ncol=len(EXP) + 1, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout()
    fs.save(fig, "cmaes_recovery_values")
    plt.close(fig)


# ---------------- Figure 3: recovery error (MAPE) of the best individual ----------------
def fig_recovery_error():
    best = {e: d.loc[d[d.experiment == e].best_nrmse.idxmin()] for e in EXP}
    fig, ax = plt.subplots(figsize=(0.72 * fs.WIDTH_FULL, 0.50 * fs.WIDTH_FULL))
    for j, p in enumerate(PARAM_ORDER):
        mapes = np.array([abs(best[e][p] - REF[p]) / abs(REF[p]) * 100 for e in EXP])
        offs = beeswarm(np.log10(np.clip(mapes, 1e-2, None)), ythr=0.085, xstep=0.05)
        for off, e, m in zip(offs, EXP, mapes):
            fs.scatter(ax, j + off, max(m, 1e-2), e, base=46)
    ax.axhline(10, color="0.55", linestyle="--", linewidth=1, alpha=0.8)
    ax.text(len(PARAM_ORDER) - 0.55, 11, "10 %", color="0.45", fontsize=8, va="bottom", ha="right")
    ax.set_yscale("log"); ax.set_xticks(range(len(PARAM_ORDER)))
    ax.set_xticklabels([PLABEL[p] for p in PARAM_ORDER], fontsize=12)
    ax.set_xlim(-0.5, len(PARAM_ORDER) - 0.5)
    ax.set_ylabel("recovery error  MAPE (%)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.legend(handles=fs.legend_handles(EXP), loc="lower center", ncol=min(len(EXP), 7),
               bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    fs.save(fig, "cmaes_recovery_error")
    plt.close(fig)


if __name__ == "__main__":
    fig_convergence()
    fig_recovery_values()
    fig_recovery_error()
