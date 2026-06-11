"""figstyle — single source of truth for the paper's figure identity.

Every figure script imports this so fonts, sizes, colours, markers and the per-station identity are
IDENTICAL across the whole paper. The Copernicus / GMD rules are documented in
Review/figure-guidelines.md; the hard ones are enforced here (one sans-serif family, embedded fonts,
>= 8 cm width, 300 dpi, colour-blind-safe AND grayscale-distinguishable via marker shape).

Station identity (FIXED everywhere): each experiment has ONE colour and ONE marker, never reused for
another station and never changed between figures. Colour encodes sea-surface temperature (cold blue
-> warm red); MERGED is the neutral joint reference (black star). The shape carries the same ranking,
so a figure stays readable in grayscale and under colour-vision deficiency.

    from seapopym_repro import figstyle as fs   # applies the house style on import
    fs.scatter(ax, x, y, "HOT")                  # HOT's fixed marker + colour + size
    handles = fs.legend_handles(exps)            # one entry per station, canonical order
    fs.save(fig, "cmaes_convergence")            # writes <stem>.pdf (vector) + <stem>.png (300 dpi)
"""
from __future__ import annotations

import matplotlib as mpl
from matplotlib.lines import Line2D

from .paths import FIGURES, ROOT, load_params  # noqa: F401  (ROOT re-exported for scripts)

# ---- GMD / Copernicus dimensions (figsize is in inches; 1 cm = 1/2.54 in) -------------------
# Copernicus: figure width must be >= 8 cm; full text width is ~17 cm. Target these, never wider.
CM = 1.0 / 2.54
WIDTH_1COL = 8.3 * CM      # single column (>= 8 cm hard minimum)
WIDTH_1HALF = 12.0 * CM    # 1.5 column
WIDTH_FULL = 17.0 * CM     # full text width (double column)

# ---- per-station identity: ONE colour + ONE marker each, fixed across every figure ----------
# colour = mean sea-surface temperature (cold -> warm); MERGED = neutral joint reference.
# `sst` (deg C) only documents the cold->warm ordering; it is not used for plotting.
STATIONS = {
    "MERGED":        dict(color="#000000", marker="*", label="MERGED",  sst=None),
    "BARENTS":       dict(color="#1A5276", marker="o", label="BARENTS", sst=1),
    "PAPA":          dict(color="#2E86C1", marker="s", label="PAPA",    sst=9),
    "Bay_of_Biscay": dict(color="#17A589", marker="^", label="BISCAY",  sst=14),
    "Canaries":      dict(color="#F39C12", marker="D", label="CANARY",  sst=19),
    "BATS":          dict(color="#E67E22", marker="v", label="BATS",    sst=21),
    "HOT":           dict(color="#C0392B", marker="p", label="HOT",     sst=24),
}
# canonical plotting order: MERGED leftmost, then single stations coldest -> warmest.
ORDER = ["MERGED", "BARENTS", "PAPA", "Bay_of_Biscay", "Canaries", "BATS", "HOT"]

# ---- model parameters (labels + reference + bounds), shared by the recovery figures ---------
_p = load_params()["model_parameters"]
REF, BOUNDS = _p["reference"], _p["bounds"]
PARAM_ORDER = ["energy_transfert", "tr_0", "gamma_tr", "lambda_temperature_0", "gamma_lambda_temperature"]
PLABEL = {
    "energy_transfert": r"$E$",
    "tr_0": r"$\tau_{r_0}$",
    "gamma_tr": r"$\gamma_{\tau_r}$",
    "lambda_temperature_0": r"$\lambda_0$",
    "gamma_lambda_temperature": r"$\gamma_\lambda$",
}


def color(exp: str) -> str:
    return STATIONS[exp]["color"]


def marker(exp: str) -> str:
    return STATIONS[exp]["marker"]


def label(exp: str) -> str:
    return STATIONS[exp]["label"]


def order(present) -> list[str]:
    """ORDER filtered to the experiments actually present (keeps the canonical order)."""
    s = set(present)
    return [e for e in ORDER if e in s]


# MERGED is emphasised everywhere (bigger star, white edge); single stations share one base size.
def _size(exp: str, base: float) -> float:
    return 3.4 * base if exp == "MERGED" else base


def scatter(ax, x, y, exp: str, base: float = 55, **kw):
    """Scatter `exp`'s points with its fixed colour + marker (MERGED emphasised)."""
    edge = "white" if exp == "MERGED" else "black"
    return ax.scatter(
        x, y, marker=marker(exp), s=_size(exp, base), color=color(exp),
        edgecolor=kw.pop("edgecolor", edge), linewidth=kw.pop("linewidth", 0.9 if exp == "MERGED" else 0.5),
        zorder=kw.pop("zorder", 6 if exp == "MERGED" else 5), **kw)


def legend_handles(present, markersize: float = 8):
    """Line2D handles (marker + colour per station) in canonical order, for fig.legend()."""
    return [Line2D([0], [0], marker=marker(e), color="none", markerfacecolor=color(e),
                   markeredgecolor=("white" if e == "MERGED" else "black"),
                   markersize=(markersize * 1.6 if e == "MERGED" else markersize),
                   linestyle="", label=label(e)) for e in order(present)]


# ---- global matplotlib style (GMD-compliant) ------------------------------------------------
def apply() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],   # one sans-serif family (text)
        "mathtext.fontset": "dejavusans",   # math variables (E, tau, gamma, lambda) italic, sans-serif, deterministic
        "font.size": 9,
        "axes.titlesize": 10, "axes.titleweight": "bold",
        "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8.5, "legend.frameon": False,
        "figure.titlesize": 11, "figure.titleweight": "bold",
        "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False, "grid.alpha": 0.3, "grid.linewidth": 0.6,
        "lines.linewidth": 1.3,
        "figure.facecolor": "white", "savefig.facecolor": "white",
        "figure.dpi": 150, "savefig.dpi": 300,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42, "ps.fonttype": 42,    # embed fonts as TrueType (Copernicus requirement)
        "svg.fonttype": "none",
    })


def save(fig, stem: str, subdir: str = "cmaes") -> None:
    """Write the figure as BOTH vector PDF (GMD primary) and 300-dpi PNG (preview/inspection)."""
    out = FIGURES / subdir if subdir else FIGURES
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        p = out / f"{stem}.{ext}"
        fig.savefig(p)
        print("wrote", p)


apply()   # a bare `import figstyle` already yields the house style
