"""Figure 8 — CMA-ES parameter recovery, two interchangeable renderings of the same data.

This script produces THREE renderings from the same seed-ensemble product (pick one with --variant, default all):
  - TABLE (stem, e.g. Figure_8): rows = experiments (MERGED + 6 stations, MERGED-first then cold->warm),
    columns = the five model parameters. Each cell shows, in black, the BEST-of-ensemble value with its
    SIGNED relative error vs the twin reference in parentheses, (best - ref) / |ref| x 100; and below in
    grey, the std across the seeded restarts (the identifiability spread). Each column header carries the
    parameter symbol and its reference value. The MERGED row is the joint recovery requested by RC-3 (#12).
  - RELATIVE-ERROR PLOT ({stem}_relerr): six panels on a shared symmetric-log axis. Panels 1-5 (one per
    parameter): the signed relative error of every seeded restart (faint points) and of the best member
    (cursor), per experiment; the reference is the 0 % line. Panel 6: the mean absolute relative error
    over the five parameters. symlog keeps the well-recovered cluster near 0 readable while the large
    equifinal errors stay on the same axis.
  - TWO HEATMAPS ({stem}_heatmaps): side by side, both expressed in % of the reference so the five
    parameters share one scale. LEFT = recovery, |best - reference|; RIGHT = restart spread, std of the
    seeds. Light = good (recovered / tight), dark = far / wide. Equifinality reads as a cell that is pale
    on the left (the best member sits on the reference) yet dark on the right (the restarts disagree).

Both tell the same story: E / lambda_0 / gamma_lambda land within ~1 % with tiny spread (identifiable);
tr_0 / gamma_tr scatter to large errors with wide spread at the warm stations (equifinal recruitment).
The table is compact for the paper; the plot exposes the per-seed dispersion.

Reads ONLY the seed-ensemble product for the production cost (paths.PRODUCTION_METRIC), frozen by
run_cmaes_seed_ensemble.py. Use --metric to render another cost's run. Compare on PARAMETER RECOVERY
vs reference, NOT on the cost (a low cost on warm stations is equifinality, not recovery).
Output : figures/{stem}.{pdf,png} (table), {stem}_relerr.{pdf,png} (plot), {stem}_heatmaps.{pdf,png}
Run    : .venv/bin/python scripts/figures/fig08_recovery.py [--metric nrmse_mean] [--variant all]
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D

from seapopym_repro import figstyle as fs, paths

ap = argparse.ArgumentParser()
ap.add_argument("--metric", default=paths.PRODUCTION_METRIC, help="cost metric whose frozen run to plot")
ap.add_argument("--variant", choices=["all", "table", "relerr", "heatmaps"], default="all",
                help="which rendering(s) to produce (default all: table + relerr plot + heatmaps)")
ap.add_argument("--stem", default="Figure_8", help="output stem; the plot is saved as <stem>_relerr")
ap.add_argument("--linthresh", type=float, default=10.0,
                help="relerr plot: symlog linear band around 0, in %% (default 10, the recovery criterion)")
ap.add_argument("--sharey", action=argparse.BooleanOptionalAction, default=True,
                help="relerr plot: share the y-axis across panels (default on)")
args = ap.parse_args()
METRIC, VARIANT, STEM, LINTHRESH, SHAREY = args.metric, args.variant, args.stem, args.linthresh, args.sharey

REF, PARAM_ORDER, PLABEL = fs.REF, fs.PARAM_ORDER, fs.PLABEL
UNIT = {                                       # compact units for the table column headers (Table 1)
    "energy_transfert": "",
    "tr_0": " d",
    "gamma_tr": " °C⁻¹",
    "lambda_temperature_0": " d⁻¹",
    "gamma_lambda_temperature": " °C⁻¹",
}

d = pd.read_csv(paths.cmaes_product("seed_ensemble", METRIC))
EXP = fs.order(set(d.experiment))                            # MERGED + present stations, canonical order
NSEEDS = int(d.groupby("experiment").seed.nunique().max())
best = d.loc[d.groupby("experiment").best_nrmse.idxmin().values].set_index("experiment")   # best seed/exp
std = d.groupby("experiment")[PARAM_ORDER].std(ddof=1)       # dispersion across the seeded restarts
print(f"metric={METRIC}, {len(EXP)} experiments, up to {NSEEDS} seeds")


def relerr(values, p):
    """Signed relative error (%) of parameter p vs its reference."""
    return (np.asarray(values, float) - REF[p]) / abs(REF[p]) * 100.0


def render_table(stem):
    """Compact recovery table: best value (signed rel. error) over std, per experiment x parameter."""
    def fmt(v):
        return f"{v:.4f}"                       # fixed 4 decimals everywhere (no scientific notation)

    def fmt_re(re):
        return f"{re:+.1f}%"                     # signed relative error, fixed 1 decimal

    # ---- layout: a hand-drawn grid (full control over the two-line, multi-colour cells) ----------
    n = len(EXP)
    LEAD = 1.7                                   # width of the leftmost (experiment-label) column
    NP = len(PARAM_ORDER)
    CELL_FS = 8.2                                # one font size for both lines of every data cell
    HEADER_Y = n + 0.5                           # header row sits in [n, n+1]
    col_x = lambda j: LEAD + j + 0.5             # centre of parameter column j
    row_y = lambda i: n - i - 0.5                # centre of experiment row i (top = first)

    fig, ax = plt.subplots(figsize=(fs.WIDTH_FULL, 0.52 * fs.WIDTH_FULL))
    ax.set_xlim(0, LEAD + NP)
    ax.set_ylim(0, n + 1)
    ax.axis("off")

    for i in range(n):                           # zebra striping for readability
        if i % 2 == 1:
            ax.axhspan(row_y(i) - 0.5, row_y(i) + 0.5, color="0.965", zorder=0)

    # rules: top frame, header/data divider (heavy), inter-row + post-label (light), bottom frame
    ax.axhline(n + 1, color="0.25", lw=1.0, zorder=3)
    ax.axhline(n, color="0.25", lw=1.1, zorder=3)
    ax.axhline(0, color="0.25", lw=1.0, zorder=3)
    for i in range(1, n):
        ax.axhline(n - i, color="0.85", lw=0.6, zorder=3)
    ax.axvline(LEAD, color="0.85", lw=0.6, zorder=3)

    # header: corner label + per-parameter (symbol over "ref = value unit")
    ax.text(0.08, HEADER_Y, "Experiment", ha="left", va="center", fontsize=8.5, fontweight="bold", color="0.3")
    for j, p in enumerate(PARAM_ORDER):
        ax.text(col_x(j), HEADER_Y + 0.14, PLABEL[p], ha="center", va="center", fontsize=13)
        ax.text(col_x(j), HEADER_Y - 0.27, f"ref = {fmt(REF[p])}{UNIT[p]}",
                ha="center", va="center", fontsize=6.8, color="0.4")

    # body: one row per experiment, one two-line cell per parameter
    for i, e in enumerate(EXP):
        yc = row_y(i)
        ax.text(0.08, yc, fs.label(e), ha="left", va="center", fontsize=CELL_FS, fontweight="bold", color=fs.color(e))
        for j, p in enumerate(PARAM_ORDER):
            v, s = best.loc[e, p], std.loc[e, p]
            re = (v - REF[p]) / abs(REF[p]) * 100.0
            ax.text(col_x(j), yc + 0.19, f"{fmt(v)} ({fmt_re(re)})", ha="center", va="center", fontsize=CELL_FS)
            ax.text(col_x(j), yc - 0.19, fmt(s), ha="center", va="center", fontsize=CELL_FS, color="0.5")

    fig.tight_layout()
    fs.save(fig, stem, subdir=None)


def render_relerr(stem):
    """Six symlog panels: signed relative error per restart + best, plus the mean absolute error."""
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
                np.abs(np.concatenate([pts[e] for e in EXP] + [[bst[e]] for e in EXP])), 99)) * 1.15) or 100.0
            for panel, (pts, bst) in pdata.items()}
    GMAX = max(pmax.values())

    fig, axes = plt.subplots(2, 3, figsize=(fs.WIDTH_FULL, 0.66 * fs.WIDTH_FULL), sharey=SHAREY)
    for ax, panel in zip(axes.flat, PANELS):
        ymax = GMAX if SHAREY else pmax[panel]
        pts, bst = pdata[panel]
        for j, e in enumerate(EXP):                                            # two sub-columns per station:
            opt_cursor(ax, x[j] - SUB, np.clip(bst[e], -ymax, ymax), e)        #  left  = optimum (cursor)
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
    fs.save(fig, stem, subdir=None)


def render_heatmaps(stem):
    """Two heatmaps side by side: recovery (|best - ref| / |ref|) and restart spread (std / |ref|), both
    in % of the reference so the five parameters share one scale. Equifinality reads as a cell that is
    WHITE on the left (the best member sits on the reference) yet dark on the right (the restarts disagree).
    Colour is binned on the 10 % recovery criterion (white = on target / tight); the signed error or the
    spread is printed in each cell, and each column header carries the parameter symbol over its reference."""
    abs_err = np.array([[abs(best.loc[e, p] - REF[p]) / abs(REF[p]) * 100.0 for p in PARAM_ORDER] for e in EXP])
    signed_err = np.array([[(best.loc[e, p] - REF[p]) / abs(REF[p]) * 100.0 for p in PARAM_ORDER] for e in EXP])
    spread = np.array([[std.loc[e, p] / abs(REF[p]) * 100.0 for p in PARAM_ORDER] for e in EXP])

    bounds = [0, 5, 10, 25, 50, 100]                       # % of reference; tied to the 10 % recovery criterion
    reds = plt.get_cmap("Reds")
    band = ["white"] + [reds(t) for t in np.linspace(0.30, 1.0, len(bounds) - 1)]   # white floor + red ramp
    cmap = ListedColormap(band[:-1])                       # 5 interior bins (white + 4 reds)
    cmap.set_over(band[-1])                                 # > 100 % = darkest red
    norm = BoundaryNorm(bounds, ncolors=cmap.N)

    def fmt_signed(v):
        if round(v, 1) == 0:
            return "0"                                       # exact recovery: drop the sign and the decimal
        return f"{v:+.0f}" if abs(v) >= 10 else f"{v:+.1f}"

    def fmt_pos(v):
        if round(v, 1) == 0:
            return "0"
        return f"{v:.0f}" if v >= 10 else f"{v:.1f}"

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(fs.WIDTH_FULL, 0.50 * fs.WIDTH_FULL),
                                   sharey=True, layout="constrained")
    panels = [(axL, abs_err, signed_err, fmt_signed, "Relative error of the best member"),
              (axR, spread, spread, fmt_pos, "Relative spread across restarts")]
    for ax, colour_mat, text_mat, fmt, title in panels:
        ax.imshow(colour_mat, cmap=cmap, norm=norm, aspect="auto")
        ax.set_xticks([])                                    # parameter headers are drawn as custom text on top
        ax.set_yticks(range(len(EXP)), [fs.label(e) for e in EXP])
        ax.set_xticks(np.arange(-0.5, len(PARAM_ORDER), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(EXP), 1), minor=True)
        ax.grid(which="minor", color="0.7", linewidth=1.0)   # grey lines stay visible over white cells
        ax.tick_params(which="both", length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        for j, p in enumerate(PARAM_ORDER):                  # header: symbol (black) over its reference (grey), no "="
            ax.text(j, 1.13, PLABEL[p], transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=11, clip_on=False)
            ax.text(j, 1.02, f"{REF[p]:.4g}", transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=7.5, color="0.5", clip_on=False)
        for i in range(len(EXP)):
            for j in range(len(PARAM_ORDER)):
                ax.text(j, i, fmt(text_mat[i, j]), ha="center", va="center", fontsize=7.5,
                        color="white" if colour_mat[i, j] >= 25 else "0.1")
        ax.set_title(title, fontsize=10, pad=42)
    axR.tick_params(labelleft=False)                       # shared y-axis: station labels on the left only
    for tl, e in zip(axL.get_yticklabels(), EXP):          # station identity = registry colour
        tl.set_color(fs.color(e))
        tl.set_fontweight("bold")

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=[axL, axR], extend="max", ticks=bounds, fraction=0.045, pad=0.02)
    cb.set_label("deviation from reference (%)", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    fs.save(fig, stem, subdir=None)


if VARIANT in ("all", "table"):
    render_table(STEM)
if VARIANT in ("all", "relerr"):
    render_relerr(f"{STEM}_relerr")
if VARIANT in ("all", "heatmaps"):
    render_heatmaps(f"{STEM}_heatmaps")
