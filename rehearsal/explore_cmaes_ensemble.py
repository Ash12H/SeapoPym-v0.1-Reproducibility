"""rehearsal/explore_cmaes_ensemble.py — EXPLORATORY figures for the seeded CMA-ES ensemble.

NOT a final manuscript figure (separate from fig0X_*). Reads the outputs of
run_cmaes_seed_ensemble.py and produces two exploratory panels for discussion:

  A. Convergence (cost function): best-so-far mean-NRMSE vs number of evaluations,
     one curve per experiment (the best seed's trajectory, the one we keep under
     narrative A = best-of-N restarts).
  B. Recovered-parameter distribution: per parameter, the 10 seeded restarts as a
     light strip + the BEST individual highlighted, per experiment, with the
     reference value and the optimisation bounds. Shows which parameters the best
     individual recovers (E, lambda0, gamma_lambda) and which stay equifinal /
     regime-dependent (tr_0, gamma_tr).

Outputs -> rehearsal/figures/exploratory/.
Run: .venv/bin/python rehearsal/explore_cmaes_ensemble.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_ga_production as ga  # shared constants/helpers (ROOT, OUT_DIR, WF, PARAM_KEYS, EXPERIMENTS)

CMA_DIR = ga.OUT_DIR / "cmaes"
CSV = ga.OUT_DIR / "cmaes_seed_ensemble.csv"
OUTDIR = ga.ROOT / "rehearsal" / "figures" / "exploratory"
LAMBDA = int(4 + 3 * np.log(5))  # DEAP default for the 5 parameters (= 8); x-axis = gen * LAMBDA

_params = yaml.safe_load(open(ga.ROOT / "parameters.yaml"))["model_parameters"]
REF = _params["reference"]
BOUNDS = _params["bounds"]

# short, readable labels in manuscript order
LABEL = {"BARENTS": "BARENTS", "PAPA": "PAPA", "Bay_of_Biscay": "Biscay", "BATS": "BATS",
         "Canaries": "Canaries", "HOT": "HOT", "MERGED": "MERGED"}
PARAM_TITLE = {
    "energy_transfert": "E  (energy transfer)",
    "lambda_temperature_0": r"$\lambda_0$  (mortality)",
    "gamma_lambda_temperature": r"$\gamma_\lambda$  (mortality T-sens.)",
    "tr_0": r"$\tau_{r0}$  (recruitment)",
    "gamma_tr": r"$\gamma_{\tau r}$  (recruitment T-sens.)",
}
# identifiable params first, then the equifinal recruitment pair
PARAM_ORDER = ["energy_transfert", "lambda_temperature_0", "gamma_lambda_temperature", "tr_0", "gamma_tr"]


def best_so_far(logbook_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Best-so-far mean-NRMSE vs cumulative evaluations, from one (best-seed) logbook."""
    df = pd.read_parquet(logbook_path)
    wf = df[ga.WF]  # Series (selecting the single MultiIndex column by tuple)
    per_gen = wf.groupby(level="Generation").max().sort_index()  # max weighted fitness per gen
    best = -per_gen.cummax().to_numpy()                                 # weighted fitness is -NRMSE
    evals = (per_gen.index.to_numpy() + 1) * LAMBDA
    return evals, best


def fig_convergence(experiments: list[str]) -> Path:
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    cmap = plt.get_cmap("turbo")
    for i, exp in enumerate(experiments):
        p = CMA_DIR / f"ga_logbook_{exp}.parquet"
        if not p.exists():
            continue
        evals, best = best_so_far(p)
        lw = 2.6 if exp == "MERGED" else 1.6
        ax.plot(evals, best, lw=lw, color=cmap(i / max(1, len(experiments) - 1)), label=LABEL[exp])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("model evaluations")
    ax.set_ylabel("best-so-far mean NRMSE")
    ax.set_title(f"CMA-ES convergence (best seed of 10, λ={LAMBDA})  —  exploratory")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(ncol=2, fontsize=9, framealpha=0.9)
    fig.tight_layout()
    out = OUTDIR / "cmaes_convergence.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_param_distribution(d: pd.DataFrame, experiments: list[str]) -> Path:
    best_idx = d.loc[d.groupby("experiment").best_nrmse.idxmin().values]  # best seed per experiment
    best_idx = best_idx.set_index("experiment")
    x = np.arange(len(experiments))

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.ravel()
    for k, param in enumerate(PARAM_ORDER):
        ax = axes[k]
        lo, hi = BOUNDS[param]
        for j, exp in enumerate(experiments):
            vals = d.loc[d.experiment == exp, param].to_numpy()
            jitter = (np.random.RandomState(0).rand(len(vals)) - 0.5) * 0.28
            ax.scatter(np.full_like(vals, x[j]) + jitter, vals, s=22, alpha=0.45,
                       color="0.45", zorder=2, label="seeds (10)" if j == 0 else None)
            if exp in best_idx.index:
                ax.scatter(x[j], best_idx.loc[exp, param], marker="*", s=170, color="crimson",
                           edgecolor="k", linewidth=0.5, zorder=4, label="best individual" if j == 0 else None)
        ax.axhline(REF[param], color="forestgreen", lw=1.6, ls="--", zorder=1, label="reference")
        ax.set_ylim(lo - 0.04 * (hi - lo), hi + 0.04 * (hi - lo))
        ax.set_xticks(x)
        ax.set_xticklabels([LABEL[e] for e in experiments], rotation=40, ha="right", fontsize=8)
        ax.set_title(PARAM_TITLE[param], fontsize=11)
        ax.grid(True, axis="y", alpha=0.25)
        if k == 0:
            ax.legend(fontsize=8, loc="upper right", framealpha=0.9)

    # 6th panel: best mean-NRMSE per experiment (the kept solution under narrative A)
    ax = axes[5]
    best_nrmse = [d.loc[d.experiment == e, "best_nrmse"].min() for e in experiments]
    ax.bar(x, best_nrmse, color="steelblue", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL[e] for e in experiments], rotation=40, ha="right", fontsize=8)
    ax.set_title("best mean NRMSE (kept solution)", fontsize=11)
    ax.grid(True, axis="y", alpha=0.25)

    fig.suptitle(f"Recovered parameters — seeded CMA-ES (λ={LAMBDA}, 10 restarts)  —  exploratory", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = OUTDIR / "cmaes_recovered_parameters.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def ape(param: str, vals) -> np.ndarray:
    """Absolute percentage error of recovered value(s) vs the reference, in %."""
    return 100.0 * np.abs(np.asarray(vals, float) - REF[param]) / abs(REF[param])


def fig_param_mape(d: pd.DataFrame, experiments: list[str]) -> Path:
    """Same layout as fig_param_distribution but on the absolute %-error (MAPE) scale."""
    best_idx = d.loc[d.groupby("experiment").best_nrmse.idxmin().values].set_index("experiment")
    x = np.arange(len(experiments))
    floor = 0.05  # % — clip so near-perfect recoveries stay visible on the log axis

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.ravel()
    for k, param in enumerate(PARAM_ORDER):
        ax = axes[k]
        for j, exp in enumerate(experiments):
            a = np.clip(ape(param, d.loc[d.experiment == exp, param].to_numpy()), floor, None)
            jitter = (np.random.RandomState(0).rand(len(a)) - 0.5) * 0.28
            ax.scatter(np.full_like(a, x[j]) + jitter, a, s=22, alpha=0.45,
                       color="0.45", zorder=2, label="seeds (10)" if j == 0 else None)
            if exp in best_idx.index:
                ba = max(float(ape(param, [best_idx.loc[exp, param]])[0]), floor)
                ax.scatter(x[j], ba, marker="*", s=170, color="crimson", edgecolor="k",
                           linewidth=0.5, zorder=4, label="best individual" if j == 0 else None)
        ax.axhline(10, color="forestgreen", lw=1.4, ls="--", zorder=1, label="10% error")
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([LABEL[e] for e in experiments], rotation=40, ha="right", fontsize=8)
        ax.set_title(PARAM_TITLE[param], fontsize=11)
        ax.set_ylabel("abs. % error")
        ax.grid(True, which="both", axis="y", alpha=0.25)
        if k == 0:
            ax.legend(fontsize=8, loc="upper left", framealpha=0.9)

    # 6th panel: MAPE of the best individual over the IDENTIFIABLE params (E, lambda0, gamma_lambda)
    # — recruitment (tr_0, gamma_tr) is known-equifinal so its % error is not a recovery-quality metric.
    ax = axes[5]
    idp = PARAM_ORDER[:3]
    mape = [float(np.mean([ape(p, [best_idx.loc[e, p]])[0] for p in idp])) for e in experiments]
    ax.bar(x, mape, color="indianred", alpha=0.85)
    ax.axhline(10, color="forestgreen", lw=1.4, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL[e] for e in experiments], rotation=40, ha="right", fontsize=8)
    ax.set_title("MAPE over identifiables E, $\\lambda_0$, $\\gamma_\\lambda$ (best)", fontsize=11)
    ax.set_ylabel("MAPE %")
    ax.grid(True, axis="y", alpha=0.25)

    fig.suptitle(f"Recovery error (MAPE) — seeded CMA-ES (λ={LAMBDA}, 10 restarts)  —  exploratory", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = OUTDIR / "cmaes_recovered_parameters_mape.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    if not CSV.exists():
        sys.exit(f"missing {CSV} — run rehearsal/run_cmaes_seed_ensemble.py first")
    d = pd.read_csv(CSV)
    experiments = [e for e in ga.EXPERIMENTS if e in set(d.experiment)]
    p1 = fig_convergence(experiments)
    p2 = fig_param_distribution(d, experiments)
    p3 = fig_param_mape(d, experiments)
    print(f"wrote:\n  {p1}\n  {p2}\n  {p3}", flush=True)


if __name__ == "__main__":
    main()
