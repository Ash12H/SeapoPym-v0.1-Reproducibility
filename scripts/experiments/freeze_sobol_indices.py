"""Sobol analysis — freeze the sensitivity indices for Figure 6 (the display contract).

Reads the raw Sobol run (results_raw/sobol/<subdir>/, produced by run_sobol_production.py) and
computes first- (S1) and total-order (ST) Sobol indices with bootstrap 95% CIs via SALib, for all
six output metrics at each station:
    RAW    : mean          variance       argmax
    ROBUST : log10(mean)    CV = std/mean  circular (cos/sin decomposition of the cyclic argmax)
All six are kept in the product as the record justifying the figure's metric choice; Figure 6 then
plots from this CSV alone (no raw parquet, no SALib at figure time).

Input  : results_raw/sobol/<subdir>/{sobol_results.parquet, run_meta.json}   (gitignored)
Output : products/sobol_indices.csv   (family, metric, station, param, S1, S1_conf, ST, ST_conf)
Run    : .venv/bin/python scripts/experiments/freeze_sobol_indices.py [--subdir conv/N8192]
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from SALib.analyze import sobol

from seapopym_repro import paths

PERIOD_DAYS = 365.0

ap = argparse.ArgumentParser()
ap.add_argument("--subdir", default="conv/N8192",
                help="raw run under results_raw/sobol/ (the converged production point)")
args = ap.parse_args()
SRC = paths.RESULTS_RAW / "sobol" / args.subdir

meta = json.loads((SRC / "run_meta.json").read_text())
NAMES = meta["names"]
PROBLEM = {"num_vars": len(NAMES), "names": NAMES, "bounds": [meta["bounds"][n] for n in NAMES]}
results = pd.read_parquet(SRC / "sobol_results.parquet").astype(float)
D = len(NAMES)
CALC2 = bool(meta.get("calc_second_order", len(results) % (2 * D + 2) == 0))
N = len(results) // ((2 * D + 2) if CALC2 else (D + 2))
STATIONS = list(results.columns.get_level_values("station").unique())
print(f"rows={len(results):,} | D={D} | calc_second_order={CALC2} | N={N:,} | stations={STATIONS}")


def analyze(y):
    out = sobol.analyze(PROBLEM, np.asarray(y, float), calc_second_order=CALC2, print_to_console=False, seed=0)
    return {k: out[k] for k in ("S1", "S1_conf", "ST", "ST_conf")}


def analyze_circular(argmax):
    """Variance-weighted S1/ST for a cyclic day-of-year variable (cos/sin decomposition)."""
    theta = 2 * np.pi * np.asarray(argmax, float) / PERIOD_DAYS
    c, s = np.cos(theta), np.sin(theta)
    rc, rs = analyze(c), analyze(s)
    wc, ws = np.var(c), np.var(s)
    w = (wc + ws) or 1.0
    return {k: (wc * rc[k] + ws * rs[k]) / w for k in ("S1", "S1_conf", "ST", "ST_conf")}


def series(st, key):
    return results[(st, key)].to_numpy(float)


RAW = {
    "mean":     lambda st: analyze(series(st, "mean")),
    "variance": lambda st: analyze(series(st, "variance")),
    "argmax":   lambda st: analyze(series(st, "argmax")),
}
ROBUST = {
    "log10(mean)": lambda st: analyze(np.log10(np.clip(series(st, "mean"), 1e-12, None))),
    "CV":          lambda st: analyze(np.sqrt(np.clip(series(st, "variance"), 0, None))
                                      / np.clip(series(st, "mean"), 1e-12, None)),
    "circular":    lambda st: analyze_circular(series(st, "argmax")),
}

rows = []
for family, fam in (("raw", RAW), ("robust", ROBUST)):
    for metric, fn in fam.items():
        for st in STATIONS:
            res = fn(st)
            for i, p in enumerate(NAMES):
                rows.append({"family": family, "metric": metric, "station": st, "param": p,
                             "S1": res["S1"][i], "S1_conf": res["S1_conf"][i],
                             "ST": res["ST"][i], "ST_conf": res["ST_conf"][i]})
paths.PRODUCTS.mkdir(parents=True, exist_ok=True)
out = paths.PRODUCTS / "sobol_indices.csv"
table = pd.DataFrame(rows).sort_values(["family", "metric", "station", "param"]).reset_index(drop=True)
table.to_csv(out, index=False)
print(f"froze Sobol indices -> {out}  ({len(rows)} rows, 6 metrics x {len(STATIONS)} stations)")
