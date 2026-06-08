"""Preliminary experiment — Sobol indices on robustified output metrics.

The published Sobol analysis uses raw `mean`, `variance`, `argmax`. Two methodological issues:
  - mean & variance are heavy-tailed (B ~ R/lambda diverges as lambda -> 0), so variance-based
    Sobol is dominated by the low-mortality tail and may *inflate* the apparent dominance of
    the mortality parameters;
  - argmax (peak day-of-year) is a *circular* variable, so ordinary variance is distorted when
    peaks fall near the year boundary.

This script RE-USES the stored Saltelli outputs (data/sobol_results.parquet) and recomputes
Sobol indices on transformed metrics — no model re-run:
  - magnitude  : mean      vs  log10(mean)
  - variability: variance  vs  CV = std / mean
  - timing     : argmax    vs  circular (variance-weighted combination of cos/sin of phase)

Output : experiment/figures/Figure_R_sobol_robust.{pdf,png} + a printed ST comparison.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from SALib.analyze import sobol


def _project_root(marker: str = "pyproject.toml") -> Path:
    for parent in [Path.cwd(), *Path.cwd().parents]:
        if (parent / marker).exists():
            return parent
    raise FileNotFoundError(f"Project root marker {marker!r} not found.")


PROJECT_ROOT = _project_root()
DATA_DIR = PROJECT_ROOT / "data"
FIG_DIR = PROJECT_ROOT / "experiment" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

with open(PROJECT_ROOT / "parameters.yaml") as f:
    SB = yaml.safe_load(f)["sobol"]

NAMES = SB["parameters_ordered"]
PROBLEM = {"num_vars": len(NAMES), "names": NAMES,
           "bounds": [SB["bounds"][n] for n in NAMES]}
PLABEL = {"energy_transfert": "E", "tr_0": "τr0", "gamma_tr": "γτr",
          "lambda_temperature_0": "λ0", "gamma_lambda_temperature": "γλ"}
STATIONS = ["HOT", "Canaries", "BATS", "Bay_of_Biscay", "PAPA", "BARENTS"]
PERIOD_DAYS = 365.0  # analysis_period spans one year

res = pd.read_parquet(DATA_DIR / "sobol_results.parquet")
n_rows = len(res)
D = PROBLEM["num_vars"]
calc_2nd = (n_rows % (2 * D + 2) == 0)  # Saltelli second-order design -> N*(2D+2)
print(f"Rows={n_rows} | D={D} | calc_second_order={calc_2nd} "
      f"(N_base={n_rows // ((2 * D + 2) if calc_2nd else (D + 2))})")


def _analyze(y):
    y = np.asarray(y, dtype=float)
    out = sobol.analyze(PROBLEM, y, calc_second_order=calc_2nd, print_to_console=False)
    return np.asarray(out["S1"]), np.asarray(out["ST"])


def _circular_timing(argmax):
    """Variance-weighted Sobol indices for a circular day-of-year variable."""
    theta = 2 * np.pi * np.asarray(argmax, dtype=float) / PERIOD_DAYS
    c, s = np.cos(theta), np.sin(theta)
    s1c, stc = _analyze(c)
    s1s, sts = _analyze(s)
    wc, ws = np.var(c), np.var(s)
    w = wc + ws if (wc + ws) > 0 else 1.0
    return (wc * s1c + ws * s1s) / w, (wc * stc + ws * sts) / w


# metric family -> (raw label, raw getter, transformed label, transformed getter)
def _get(st, m):
    return res[(st, m)].values


FAMILIES = {
    "magnitude":   ("mean",     lambda st: _get(st, "mean"),
                    "log10(mean)", lambda st: np.log10(np.clip(_get(st, "mean"), 1e-12, None))),
    "variability": ("variance", lambda st: _get(st, "variance"),
                    "CV",        lambda st: np.sqrt(np.clip(_get(st, "variance"), 0, None)) / np.clip(_get(st, "mean"), 1e-12, None)),
    "timing":      ("argmax",   lambda st: _get(st, "argmax"),
                    "circular",  None),  # handled specially
}

# Collect ST (and S1) for every family / variant / station / parameter.
rows = []
for fam, (raw_lbl, raw_fn, tr_lbl, tr_fn) in FAMILIES.items():
    for st in STATIONS:
        s1_raw, st_raw = _analyze(raw_fn(st))
        if fam == "timing":
            s1_tr, st_tr = _circular_timing(_get(st, "argmax"))
        else:
            s1_tr, st_tr = _analyze(tr_fn(st))
        for i, p in enumerate(NAMES):
            rows.append({"family": fam, "station": st, "param": PLABEL[p],
                         "ST_raw": st_raw[i], "ST_tr": st_tr[i],
                         "S1_raw": s1_raw[i], "S1_tr": s1_tr[i]})
df = pd.DataFrame(rows)

# ---- Printed comparison: ST averaged across stations, raw vs transformed ----
print("\n=== ST (mean across 6 stations) — raw vs transformed metric ===")
for fam in FAMILIES:
    sub = df[df.family == fam].groupby("param")[["ST_raw", "ST_tr"]].mean()
    sub = sub.reindex([PLABEL[p] for p in NAMES])
    raw_lbl = FAMILIES[fam][0]
    tr_lbl = FAMILIES[fam][2]
    print(f"\n[{fam}]  {raw_lbl} -> {tr_lbl}")
    print(f"{'param':>6s}{'ST_raw':>10s}{'ST_tr':>10s}{'Δ':>9s}")
    for p, r in sub.iterrows():
        print(f"{p:>6s}{r.ST_raw:10.3f}{r.ST_tr:10.3f}{r.ST_tr - r.ST_raw:+9.3f}")
    dom_raw = sub.ST_raw.idxmax()
    dom_tr = sub.ST_tr.idxmax()
    print(f"  dominant param: raw={dom_raw}  ->  transformed={dom_tr}")


# ---- Figure: ST per parameter, raw vs transformed, one panel per family ----
fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
x = np.arange(len(NAMES))
w = 0.38
for ax, fam in zip(axes, FAMILIES):
    sub = df[df.family == fam].groupby("param")[["ST_raw", "ST_tr"]].agg(["mean", "std"])
    sub = sub.reindex([PLABEL[p] for p in NAMES])
    raw_m, raw_s = sub[("ST_raw", "mean")].values, sub[("ST_raw", "std")].values
    tr_m, tr_s = sub[("ST_tr", "mean")].values, sub[("ST_tr", "std")].values
    ax.bar(x - w / 2, raw_m, w, yerr=raw_s, capsize=3, label=f"raw ({FAMILIES[fam][0]})",
           color="#9ecae1", edgecolor="k", linewidth=0.4)
    ax.bar(x + w / 2, tr_m, w, yerr=tr_s, capsize=3, label=f"transformed ({FAMILIES[fam][2]})",
           color="#fc9272", edgecolor="k", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([PLABEL[p] for p in NAMES])
    ax.set_title(fam, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)
axes[0].set_ylabel(r"$S_T$ (mean ± std across 6 stations)")
fig.suptitle("Sobol total-order index $S_T$: raw vs robustified output metrics (test mode)",
             fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.96])
out_pdf = FIG_DIR / "Figure_R_sobol_robust.pdf"
out_png = FIG_DIR / "Figure_R_sobol_robust.png"
fig.savefig(out_pdf, bbox_inches="tight")
fig.savefig(out_png, dpi=200, bbox_inches="tight", facecolor="white")
print(f"\nSaved {out_pdf}")
print(f"Saved {out_png}")
