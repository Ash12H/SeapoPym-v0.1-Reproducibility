"""Figure 8 — CMA-ES parameter recovery: relative error vs the twin reference.

Six panels on a SHARED symmetric-log axis. Panels 1-5 (one per parameter): the SIGNED relative error
(recovered - reference) / reference x 100 of every seeded restart (faint points) and of the best-of-
ensemble member (fixed marker), per experiment; the reference is the 0 % line, so identifiability reads
directly off the vertical spread around zero and the sign shows over- vs under-estimation. E / lambda_0
/ gamma_lambda collapse onto 0 % (identifiable); tr_0 / gamma_tr scatter to +/- hundreds of % at the
warm stations (equifinal recruitment). Panel 6: the MEAN ABSOLUTE relative error over the five
parameters (a magnitude, positive only) — the compact "how far off overall" summary per experiment.

symlog (linthresh = 1 %): linear within +/- 1 % so the well-recovered cluster near 0 is not magnified,
logarithmic beyond so the large equifinal errors stay on the same axis. Station identity = colour +
marker + the shared legend (no x ticks).

Reads ONLY the seed-ensemble product for the production cost (paths.PRODUCTION_METRIC), frozen by
run_cmaes_seed_ensemble.py. Use --metric to render another cost's run. Compare on PARAMETER RECOVERY
vs reference, NOT on the cost (a low cost on warm stations is equifinality, not recovery).
Output : figures/Figure_8.{pdf,png}
Run    : .venv/bin/python scripts/figures/fig08_recovery.py [--metric nrmse_mean]
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from seapopym_repro import figstyle as fs, paths

ap = argparse.ArgumentParser()
ap.add_argument("--metric", default=paths.PRODUCTION_METRIC, help="cost metric whose frozen run to plot")
ap.add_argument("--linthresh", type=float, default=10.0, help="symlog linear band around 0, in %% (default 10, the recovery criterion)")
ap.add_argument("--sharey", action=argparse.BooleanOptionalAction, default=True,
                help="share the y-axis across panels (default on); --no-sharey gives each panel its own scale")
ap.add_argument("--stem", default="Figure_8", help="output filename stem (default Figure_8)")
args = ap.parse_args()
METRIC, LINTHRESH, SHAREY, STEM = args.metric, args.linthresh, args.sharey, args.stem

REF, PARAM_ORDER, PLABEL = fs.REF, fs.PARAM_ORDER, fs.PLABEL

d = pd.read_csv(paths.cmaes_product("seed_ensemble", METRIC))
EXP = fs.order(set(d.experiment))                            # canonical order, only what's present
NSEEDS = int(d.groupby("experiment").seed.nunique().max())
print(f"metric={METRIC}, {len(EXP)} experiments, up to {NSEEDS} seeds")


def relerr(values, p):
    """Signed relative error (%) of parameter p vs its reference."""
    return (np.asarray(values, float) - REF[p]) / abs(REF[p]) * 100.0


# per-seed signed relative error per parameter, and per-seed mean ABSOLUTE relative error (panel 6)
seed_re = {p: {e: relerr(d.loc[d.experiment == e, p].to_numpy(), p) for e in EXP} for p in PARAM_ORDER}
seed_meanabs = {e: np.mean([np.abs(relerr(d.loc[d.experiment == e, p].to_numpy(), p)) for p in PARAM_ORDER], axis=0)
                for e in EXP}
best_row = d.loc[d.groupby("experiment").best_nrmse.idxmin().values].set_index("experiment")
best_re = {p: {e: relerr([best_row.loc[e, p]], p)[0] for e in EXP} for p in PARAM_ORDER}
best_meanabs = {e: float(np.mean([abs(best_re[p][e]) for p in PARAM_ORDER])) for e in EXP}

x = np.arange(len(EXP))
SUB = 0.2                                            # half-spacing of the two sub-columns within a station
jit = (np.random.RandomState(0).rand(NSEEDS) - 0.5) * 0.24   # per-seed jitter within the points sub-column
# display order: identifiable trio (E, lambda_0, gamma_lambda) first, then equifinal recruitment, then summary
PANELS = ["energy_transfert", "lambda_temperature_0", "gamma_lambda_temperature", "tr_0", "gamma_tr", "__mean__"]
TITLES = {**PLABEL, "__mean__": "Mean absolute\npercentage error"}   # mean |relative error| over the 5 params


def panel_points(panel):
    """(per-seed arrays, best value) per experiment for a panel; signed for params, |.| for the mean."""
    if panel == "__mean__":
        return {e: seed_meanabs[e] for e in EXP}, {e: best_meanabs[e] for e in EXP}
    return {e: seed_re[panel][e] for e in EXP}, {e: best_re[panel][e] for e in EXP}


def opt_cursor(ax, xx, yy, e):
    """Best parameter as a small right-pointing triangle (a cursor marking the precise value); colour = station."""
    big = e == "MERGED"
    ax.scatter(xx, yy, marker=">", s=40 if big else 28, color=fs.color(e),
               edgecolor="black", linewidth=0.4, zorder=7 if big else 6)


# per-panel symmetric y-limit (99th pct, robust to a lone wild seed); the global max is used when sharey
pdata = {panel: panel_points(panel) for panel in PANELS}
pmax = {panel: (float(np.nanpercentile(
            np.abs(np.concatenate([pts[e] for e in EXP] + [[best[e]] for e in EXP])), 99)) * 1.15) or 100.0
        for panel, (pts, best) in pdata.items()}
GMAX = max(pmax.values())

fig, axes = plt.subplots(2, 3, figsize=(fs.WIDTH_FULL, 0.66 * fs.WIDTH_FULL), sharey=SHAREY)
for ax, panel in zip(axes.flat, PANELS):
    ymax = GMAX if SHAREY else pmax[panel]
    pts, best = pdata[panel]
    for j, e in enumerate(EXP):                                            # two sub-columns per station:
        opt_cursor(ax, x[j] - SUB, np.clip(best[e], -ymax, ymax), e)       #  left  = optimum (cursor)
        ax.scatter(x[j] + SUB + jit[:len(pts[e])], np.clip(pts[e], -ymax, ymax), s=11, alpha=0.40,
                   color=fs.color(e), edgecolor="none", zorder=3)          #  right = per-seed restarts
    ax.set_yscale("symlog", linthresh=LINTHRESH)
    ax.set_ylim(-ymax, ymax)
    ax.set_xlim(-0.6, len(EXP) - 0.4)
    ax.set_xticks([])                              # station identity = colour + the shared legend
    ax.set_title(TITLES[panel], fontsize=9.5 if panel == "__mean__" else 12)
    ax.grid(True, axis="y", which="both", alpha=0.25)
ylabel_axes = axes[:, 0] if SHAREY else axes.flat   # per-panel scales -> label every panel
for ax in ylabel_axes:
    ax.set_ylabel("relative error (%)")

# horizontal framed legend below; station identity = colour of the ▶ cursor
station_handles = [Line2D([0], [0], marker=">", color="none", markerfacecolor=fs.color(e),
                          markeredgecolor="black", markersize=8 if e == "MERGED" else 7,
                          linestyle="", label=fs.label(e)) for e in EXP]
extra = [Line2D([0], [0], marker="o", color="none", markerfacecolor="0.5", markersize=6, linestyle="",
                label="per-seed restart")]
# 8 entries as 2 rows x 4 (matplotlib packs column-major, so reorder to keep the canonical cold->warm
# reading across rows: top MERGED/BARENTS/PAPA/BISCAY, bottom CANARY/BATS/HOT/per-seed).
allh = station_handles + extra
NCOL = 4
top, bot = allh[:NCOL], allh[NCOL:]
ordered = [h for i in range(NCOL) for h in ([top[i]] + ([bot[i]] if i < len(bot) else []))]
fig.tight_layout(rect=[0, 0.12, 1, 1])   # reserve a bottom band so the 2-row legend doesn't overlap panels
fig.legend(handles=ordered, loc="lower center", ncol=NCOL, bbox_to_anchor=(0.5, 0.0), frameon=True)
fs.save(fig, STEM, subdir=None)
