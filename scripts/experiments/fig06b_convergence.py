"""rehearsal/fig06b_convergence.py — Sobol convergence study (mechanical N selection).

Standard GSA convergence criterion (Sarrazin, Pianosi & Wagener, 2016, Environmental
Modelling & Software): pick the smallest sufficient N **mechanically** (no eyeballing).

    chosen N* = smallest N such that, across the REPORTED metrics x station x param:
        (a) max bootstrap 95% CI half-width of S1 and ST   <= CI_THRESHOLD   (default 0.05)
        (b) max |index(N) - index(previous N)|  (S1 and ST) <= STAB_THRESHOLD (default 0.05)

The decision is taken on the metrics we actually REPORT = the 3 ROBUST ones
(log10(mean), CV, circular). The 3 RAW heavy-tailed metrics (mean, variance, argmax) are
still computed and shown for context, but they are NOT used to gate N: the raw `variance`
Sobol indices barely converge even at large N (heavy tail B~R/lambda) — which is exactly
the methodological reason for using robust metrics.

Importable: `load_tables()`, `convergence_frame()`, `indices_table()` are reused by the
adaptive orchestrator run_sobol_convergence.py.

Outputs (rehearsal/sobol/conv/):
    convergence_metrics.csv      N, total_sims, ci_reported, ci_raw, delta_reported, converged
    Figure_6b_convergence.png    ci (robust & raw) and delta vs total simulations, with threshold
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from SALib.analyze import sobol

from seapopym_repro import paths
CONV = paths.RESULTS_RAW / "sobol" / "conv"
PERIOD_DAYS = 365.0
STATIONS_ORDER = ["BARENTS", "PAPA", "Bay_of_Biscay", "Canaries", "BATS", "HOT"]
REPORTED = ["log10(mean)", "CV", "circular"]   # metrics used for the convergence decision
RAW = ["mean", "variance", "argmax"]           # kept for context, NOT used to gate N


def analyze(problem, calc2, y):
    # seed the bootstrap so the CI half-widths (hence the convergence decision) are reproducible
    o = sobol.analyze(problem, np.asarray(y, float), calc_second_order=calc2, print_to_console=False, seed=0)
    return {"S1": o["S1"], "S1_conf": o["S1_conf"], "ST": o["ST"], "ST_conf": o["ST_conf"]}


def analyze_circular(problem, calc2, argmax):
    th = 2 * np.pi * np.asarray(argmax, float) / PERIOD_DAYS
    c, s = np.cos(th), np.sin(th)
    rc, rs = analyze(problem, calc2, c), analyze(problem, calc2, s)
    wc, ws = np.var(c), np.var(s)
    w = (wc + ws) or 1.0
    return {k: (wc * rc[k] + ws * rs[k]) / w for k in ("S1", "S1_conf", "ST", "ST_conf")}


def _complete(d: Path) -> bool:
    rp, mp = d / "sobol_results.parquet", d / "run_meta.json"
    if not (rp.exists() and mp.exists()):
        return False
    return len(pd.read_parquet(rp)) >= json.loads(mp.read_text())["total_samples"]


def indices_table(point_dir: Path) -> pd.DataFrame:
    """Tidy (metric, station, param) -> S1,S1_conf,ST,ST_conf for one completed sweep point."""
    meta = json.loads((point_dir / "run_meta.json").read_text())
    names = meta["names"]
    problem = {"num_vars": len(names), "names": names, "bounds": [meta["bounds"][n] for n in names]}
    calc2 = bool(meta.get("calc_second_order", False))
    res = pd.read_parquet(point_dir / "sobol_results.parquet").astype(float)
    stations = [s for s in STATIONS_ORDER if s in res.columns.get_level_values("station")]

    def col(st, key):
        return res[(st, key)].to_numpy(float)

    fam = {
        "mean":        lambda st: analyze(problem, calc2, col(st, "mean")),
        "variance":    lambda st: analyze(problem, calc2, col(st, "variance")),
        "argmax":      lambda st: analyze(problem, calc2, col(st, "argmax")),
        "log10(mean)": lambda st: analyze(problem, calc2, np.log10(np.clip(col(st, "mean"), 1e-12, None))),
        "CV":          lambda st: analyze(problem, calc2, np.sqrt(np.clip(col(st, "variance"), 0, None))
                                          / np.clip(col(st, "mean"), 1e-12, None)),
        "circular":    lambda st: analyze_circular(problem, calc2, col(st, "argmax")),
    }
    rows = []
    for metric, fn in fam.items():
        for st in stations:
            r = fn(st)
            for i, p in enumerate(names):
                rows.append({"metric": metric, "station": st, "param": p,
                             "S1": r["S1"][i], "S1_conf": r["S1_conf"][i],
                             "ST": r["ST"][i], "ST_conf": r["ST_conf"][i]})
    df = pd.DataFrame(rows)
    df["N"] = meta["n_samples_per_dim"]
    df["total_sims"] = meta["total_samples"]
    return df


def load_tables() -> dict[int, pd.DataFrame]:
    pts = [d for d in CONV.glob("N*") if _complete(d)]
    return {int(t.N.iloc[0]): t for t in (indices_table(d) for d in pts)}


def convergence_frame(tables: dict[int, pd.DataFrame], ci: float, stab: float) -> pd.DataFrame:
    rows, prev = [], None
    for N in sorted(tables):
        t = tables[N]
        rep = t[t.metric.isin(REPORTED)]
        raw = t[t.metric.isin(RAW)]
        ci_rep = float(max(rep.S1_conf.max(), rep.ST_conf.max()))
        ci_raw = float(max(raw.S1_conf.max(), raw.ST_conf.max()))
        if prev is None:
            delta = np.nan
        else:
            pr = tables[prev]
            m = rep.merge(pr[pr.metric.isin(REPORTED)], on=["metric", "station", "param"], suffixes=("", "_p"))
            delta = float(np.maximum((m.S1 - m.S1_p).abs(), (m.ST - m.ST_p).abs()).max())
        converged = (ci_rep <= ci) and np.isfinite(delta) and (delta <= stab)
        rows.append({"N": N, "total_sims": int(t.total_sims.iloc[0]),
                     "ci_reported": ci_rep, "ci_raw": ci_raw, "delta_reported": delta,
                     "converged": converged})
        prev = N
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ci", type=float, default=0.05, help="max bootstrap CI half-width threshold")
    ap.add_argument("--stab", type=float, default=0.05, help="max index change vs previous N threshold")
    args = ap.parse_args()

    tables = load_tables()
    if not tables:
        raise SystemExit(f"No completed sweep points under {CONV}.")
    print("completed points:", [f"N{n}" for n in sorted(tables)])
    conv = convergence_frame(tables, args.ci, args.stab)
    CONV.mkdir(parents=True, exist_ok=True)
    conv.to_csv(CONV / "convergence_metrics.csv", index=False)
    print("\n=== convergence (decision on REPORTED robust metrics) ===")
    print(conv.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    chosen = conv[conv.converged]
    if len(chosen):
        r = chosen.iloc[0]
        print(f"\nMECHANICAL DECISION (CI<={args.ci}, stab<={args.stab}): "
              f"N* = {int(r.N):,} -> {int(r.total_sims):,} simulations")
    else:
        print(f"\nNOT YET CONVERGED (CI<={args.ci}, stab<={args.stab}) — extend the sweep.")

    fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
    ax.plot(conv.total_sims, conv.ci_reported, "o-", color="#d62728", label="max CI half-width — robust (reported)")
    ax.plot(conv.total_sims, conv.ci_raw, "^:", color="gray", label="max CI half-width — raw (context, not gated)")
    ax.plot(conv.total_sims, conv.delta_reported, "s--", color="#1f77b4", label="max |Δindex| vs previous N — robust")
    ax.axhline(args.ci, color="green", ls=":", lw=1.3, label=f"threshold = {args.ci}")
    ax.set_xscale("log")
    ax.set_xlabel("total simulations  (N × (D+2))")
    ax.set_ylabel("convergence metric")
    ax.set_title("Sobol convergence (Sarrazin et al. 2016) — decision on robust metrics", fontweight="bold", fontsize=11)
    if len(chosen):
        xs = int(chosen.iloc[0].total_sims)
        ax.axvline(xs, color="green", lw=1.4, alpha=0.8)
        ax.text(xs, ax.get_ylim()[1] * 0.88, f"  N*={int(chosen.iloc[0].N):,}",
                color="green", fontsize=10, fontweight="bold")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()
    out = CONV / "Figure_6b_convergence.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"\nwrote {CONV / 'convergence_metrics.csv'}\nwrote {out}")


if __name__ == "__main__":
    main()
