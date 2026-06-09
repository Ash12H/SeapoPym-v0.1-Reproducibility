"""rehearsal/fig06_sobol.py — Figure 6: Sobol sensitivity (implicit solver, unified GA bounds).

Reads the rehearsal Sobol run produced by run_sobol_production.py:
    rehearsal/sobol/sobol_results.parquet   (per-sample mean, variance, argmax per station)
    rehearsal/sobol/run_meta.json           (parameter names, unified GA bounds, metrics)

Computes Sobol first-order (S1) and total-order (ST) indices with bootstrap 95% CIs for
ALL SIX output metrics at each of the 6 stations, and saves the full table so any metric can
be chosen — or compared — for the paper:
    RAW    : mean          variance       argmax        (the published metrics)
    ROBUST : log10(mean)    CV = std/mean  circular      (heavy-tail / circular corrections)

The 3 raw quantities are stored; the 3 robust are exact transforms of them (nothing lost):
    log10(mean) <- mean ;  CV <- sqrt(variance)/mean ;  circular <- argmax (day-of-year, cyclic).

Outputs (rehearsal/sobol/):
    sobol_indices.csv   tidy table: family,metric,station,param,S1,S1_conf,ST,ST_conf
    Figure_6.png        ROBUST metrics  (3 rows x 5 params, 6 stations on x), S1+ST bars + CI
    Figure_6_raw.png    RAW metrics     (same layout) — for comparison / paper choice

Works on any sobol_results.parquet (test 12,288 rows or production 1,190,700): the sample size
N and the second-order design flag are inferred from the row count.

Run:
    .venv/bin/python rehearsal/fig06_sobol.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import yaml
from SALib.analyze import sobol

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SOB = ROOT / "rehearsal" / "sobol"

_ap = argparse.ArgumentParser()
_ap.add_argument("--subdir", default=None,
                 help="read run + write figure under rehearsal/sobol/<subdir>/ (e.g. conv/N16384)")
_args = _ap.parse_args()
SRC = SOB / _args.subdir if _args.subdir else SOB
FIGURES = ROOT / "rehearsal" / "figures"   # canonical manuscript-figure location (like Figure_3/5/7…)
FIGURES.mkdir(parents=True, exist_ok=True)

PLABEL = {"energy_transfert": r"$E$", "tr_0": r"$\tau_{r_0}$", "gamma_tr": r"$\gamma_{\tau_r}$",
          "lambda_temperature_0": r"$\lambda_0$", "gamma_lambda_temperature": r"$\gamma_\lambda$"}
PERIOD_DAYS = 365.0
C_S1, C_ST = "#1f77b4", "#ff7f0e"

# --- load run + problem definition (bounds/names from the run's own meta) ----------------------
meta = json.loads((SRC / "run_meta.json").read_text())
NAMES = meta["names"]
PROBLEM = {"num_vars": len(NAMES), "names": NAMES, "bounds": [meta["bounds"][n] for n in NAMES]}
results = pd.read_parquet(SRC / "sobol_results.parquet").astype(float)
D = len(NAMES)
CALC2 = bool(meta.get("calc_second_order", len(results) % (2 * D + 2) == 0))
N = len(results) // ((2 * D + 2) if CALC2 else (D + 2))
print(f"rows={len(results):,} | D={D} | calc_second_order={CALC2} | N={N:,}")

# --- station order: coldest -> warmest (manuscript convention) --------------------------------
period = meta["analysis_period"]
Tref = yaml.safe_load(open(ROOT / "parameters.yaml"))["model_parameters"]["T_ref"]
st_raw = xr.open_zarr(DATA / "stations.zarr").sel(time=slice(period["start"], period["end"]))
Teff_mean = np.maximum(st_raw.temperature, Tref).mean("time").to_pandas()
present = [s for s in results.columns.get_level_values("station").unique()]
STATIONS = sorted(present, key=lambda s: float(Teff_mean[s]))
print("stations (cold->warm):", STATIONS)


def analyze(y: np.ndarray) -> dict:
    out = sobol.analyze(PROBLEM, np.asarray(y, float), calc_second_order=CALC2, print_to_console=False, seed=0)
    return {"S1": out["S1"], "S1_conf": out["S1_conf"], "ST": out["ST"], "ST_conf": out["ST_conf"]}


def analyze_circular(argmax: np.ndarray) -> dict:
    """Variance-weighted S1/ST for a cyclic day-of-year variable (cos/sin decomposition)."""
    theta = 2 * np.pi * np.asarray(argmax, float) / PERIOD_DAYS
    c, s = np.cos(theta), np.sin(theta)
    rc, rs = analyze(c), analyze(s)
    wc, ws = np.var(c), np.var(s)
    w = (wc + ws) or 1.0
    return {k: (wc * rc[k] + ws * rs[k]) / w for k in ("S1", "S1_conf", "ST", "ST_conf")}


def metric_series(station: str, key: str) -> np.ndarray:
    return results[(station, key)].to_numpy(float)


# family -> {metric_label: callable(station) -> indices dict}
RAW = {
    "mean":     lambda st: analyze(metric_series(st, "mean")),
    "variance": lambda st: analyze(metric_series(st, "variance")),
    "argmax":   lambda st: analyze(metric_series(st, "argmax")),
}
ROBUST = {
    "log10(mean)": lambda st: analyze(np.log10(np.clip(metric_series(st, "mean"), 1e-12, None))),
    "CV":          lambda st: analyze(np.sqrt(np.clip(metric_series(st, "variance"), 0, None))
                                      / np.clip(metric_series(st, "mean"), 1e-12, None)),
    "circular":    lambda st: analyze_circular(metric_series(st, "argmax")),
}

# --- compute every index once, store tidy -----------------------------------------------------
rows = []
cache: dict[tuple[str, str], dict] = {}
for family, fam in (("raw", RAW), ("robust", ROBUST)):
    for metric, fn in fam.items():
        for st in STATIONS:
            res = fn(st)
            cache[(metric, st)] = res
            for i, p in enumerate(NAMES):
                rows.append({"family": family, "metric": metric, "station": st, "param": p,
                             "S1": res["S1"][i], "S1_conf": res["S1_conf"][i],
                             "ST": res["ST"][i], "ST_conf": res["ST_conf"][i]})
table = pd.DataFrame(rows)
table.to_csv(SRC / "sobol_indices.csv", index=False)
print(f"wrote {SRC / 'sobol_indices.csv'}  ({len(table)} rows, all 6 metrics kept)")


# --- plotting: 3 metric rows x 5 param cols; 6 stations on x; S1 (blue) + ST (orange) bars -----
def make_figure(metrics: list[str], title: str, out: Path) -> None:
    nrow, ncol = len(metrics), len(NAMES)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.5 * ncol, 2.4 * nrow),
                             sharex=True, sharey=True, squeeze=False)
    x = np.arange(len(STATIONS))
    w = 0.38
    for r, metric in enumerate(metrics):
        for c, p in enumerate(NAMES):
            ax = axes[r][c]
            s1 = np.array([cache[(metric, st)]["S1"][c] for st in STATIONS])
            s1c = np.array([cache[(metric, st)]["S1_conf"][c] for st in STATIONS])
            st_ = np.array([cache[(metric, st)]["ST"][c] for st in STATIONS])
            stc = np.array([cache[(metric, st)]["ST_conf"][c] for st in STATIONS])
            ax.bar(x - w / 2, s1, w, yerr=s1c, capsize=2, color=C_S1, edgecolor="k", linewidth=0.3,
                   error_kw={"linewidth": 0.6}, label="$S_1$")
            ax.bar(x + w / 2, st_, w, yerr=stc, capsize=2, color=C_ST, edgecolor="k", linewidth=0.3,
                   error_kw={"linewidth": 0.6}, label="$S_T$")
            ax.axhline(0, color="gray", linewidth=0.5)
            if r == 0:
                ax.set_title(PLABEL[p], fontsize=13)
            if c == 0:
                ax.set_ylabel(metric, fontsize=10, fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels([s[:4] for s in STATIONS], rotation=60, fontsize=7)
            ax.grid(True, axis="y", alpha=0.25)
    axes[0][0].set_ylim(-0.05, 1.05)
    axes[0][-1].legend(fontsize=9, loc="upper right")
    fig.suptitle(title, fontweight="bold", fontsize=12)
    fig.text(0.5, 0.01, "stations (cold → warm)", ha="center", fontsize=9)
    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}")


tag = f"N={N:,}" + ("" if CALC2 else " (1st-order design)")
# Canonical manuscript figure -> rehearsal/figures/ (with the others); raw comparison stays with the data.
make_figure(list(ROBUST), f"Sobol sensitivity — robust metrics (implicit solver, GA bounds, {tag})",
            FIGURES / "Figure_6.png")
make_figure(list(RAW), f"Sobol sensitivity — raw metrics (implicit solver, GA bounds, {tag})",
            SRC / "Figure_6_raw.png")
