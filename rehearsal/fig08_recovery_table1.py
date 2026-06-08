"""fig7_and_table1.py — REHEARSAL Figure 7 (A+B) and Table 1 (optimised columns).

Reads the best individual of each logbook in rehearsal/logbooks/ and produces:
  * Figure_7A : recovery error  MAPE = |opt - ref| / |ref| * 100  (beeswarm, log y)
  * Figure_7B : parameter value within the CURRENT bounds (parameters.yaml), ref marked
  * table1_optimised.csv : best parameters per experiment (+ reference row)

Bounds and reference are read from parameters.yaml (NOT hard-coded) so Fig 7B is
correct for the current optimisation bounds. Renders whatever logbooks are present.

Output: rehearsal/figures/Figure_8_recovery_error.png, Figure_8.png, rehearsal/logbooks/table1_optimised.csv
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
LOGS = HERE / "logbooks"
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)
WF = ("Weighted_fitness", "Weighted_fitness")

ORDER = ["BARENTS", "PAPA", "Bay_of_Biscay", "BATS", "Canaries", "HOT", "MERGED"]
DISPLAY = {"BARENTS": "BARENTS", "PAPA": "PAPA", "Bay_of_Biscay": "BISCAY",
           "BATS": "BATS", "Canaries": "CANARY", "HOT": "HOT", "MERGED": "MERGED"}
PARAM_ORDER = ["energy_transfert", "tr_0", "gamma_tr", "lambda_temperature_0", "gamma_lambda_temperature"]
LABELS = {"energy_transfert": r"$E$", "tr_0": r"$\tau_{r_0}$", "gamma_tr": r"$\gamma_{\tau_r}$",
          "lambda_temperature_0": r"$\lambda_0$", "gamma_lambda_temperature": r"$\gamma_\lambda$"}

_p = yaml.safe_load(open(ROOT / "parameters.yaml"))["model_parameters"]
REF = _p["reference"]
BOUNDS = _p["bounds"]  # CURRENT optimisation bounds

logs = {p.stem.replace("ga_logbook_", ""): p for p in sorted(LOGS.glob("ga_logbook_*.parquet"))}
EXP = [e for e in ORDER if e in logs] + [e for e in logs if e not in ORDER]
if not EXP:
    raise SystemExit(f"No logbooks found in {LOGS}")
print("experiments present:", EXP)

best = {e: pd.read_parquet(logs[e]).loc[lambda d: d[WF].idxmax(), "Parametre"].to_dict() for e in EXP}
colors = dict(zip(EXP, plt.cm.tab10(np.linspace(0, 1, max(len(EXP), 1)))))
if "MERGED" in colors:
    colors["MERGED"] = (0.1, 0.1, 0.1, 1.0)


def _style(e):
    if e == "MERGED":
        return dict(marker="*", s=340, color=colors[e], edgecolor="white", linewidth=0.9, zorder=6)
    return dict(marker="o", s=95, color=colors[e], edgecolor="black", linewidth=0.5, zorder=5)


def beeswarm(yvals, ythr, xstep, max_lane=8):
    yvals = np.asarray(yvals, dtype=float)
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


# ---------------- Figure 7A: recovery MAPE (log y) ----------------
fig, ax = plt.subplots(figsize=(9, 5.2), dpi=200)
for j, p in enumerate(PARAM_ORDER):
    mapes = np.array([abs(best[e][p] - REF[p]) / abs(REF[p]) * 100 for e in EXP])
    offs = beeswarm(np.log10(np.clip(mapes, 1e-2, None)), ythr=0.085, xstep=0.05)
    for off, e, m in zip(offs, EXP, mapes):
        ax.scatter(j + off, max(m, 1e-2), **_style(e))
ax.axhline(10, color="gray", linestyle="--", linewidth=1, alpha=0.7)
ax.text(len(PARAM_ORDER) - 0.55, 10.6, "10 %", color="gray", fontsize=9, va="bottom", ha="right")
ax.set_yscale("log")
ax.set_xticks(range(len(PARAM_ORDER)))
ax.set_xticklabels([LABELS[p] for p in PARAM_ORDER], fontsize=13)
ax.set_xlim(-0.5, len(PARAM_ORDER) - 0.5)
ax.set_ylabel("Recovery error  MAPE (%)")
ax.set_title("Twin experiments — parameter recovery of the best individual (SBX, pop 256)",
             fontweight="bold", fontsize=11)
ax.grid(True, axis="y", alpha=0.3)
handles = [Line2D([0], [0], marker="*" if e == "MERGED" else "o", color="w", markerfacecolor=colors[e],
                  markeredgecolor="black", markersize=15 if e == "MERGED" else 9, linestyle="",
                  label=DISPLAY.get(e, e)) for e in EXP]
fig.legend(handles=handles, loc="lower center", ncol=min(len(EXP), 7), frameon=False,
           bbox_to_anchor=(0.5, -0.02), fontsize=9)
plt.tight_layout(rect=[0, 0.05, 1, 1])
fig.savefig(FIG / "Figure_8_recovery_error.png", dpi=200, bbox_inches="tight", facecolor="white")
print("wrote", FIG / "Figure_8_recovery_error.png")

# ---------------- Figure 7B: value within (current) bounds ----------------
fig, axes = plt.subplots(1, 5, figsize=(13, 4.6), dpi=200)
for ax, p in zip(axes, PARAM_ORDER):
    lo, hi = BOUNDS[p]
    vals = np.array([best[e][p] for e in EXP])
    offs = beeswarm(vals, ythr=(hi - lo) * 0.05, xstep=0.075)
    for off, e, v in zip(offs, EXP, vals):
        ax.scatter(off, v, **_style(e))
    ax.axhline(REF[p], color="gray", linestyle="--", linewidth=1.3)
    ax.set_ylim(lo, hi)
    ax.set_xlim(-0.6, 0.6)
    ax.set_xticks([])
    ax.set_title(LABELS[p], fontsize=13)
    ax.grid(True, axis="y", alpha=0.3)
axes[0].set_ylabel("Parameter value (within bounds)")
handles2 = handles + [Line2D([0], [0], color="gray", linestyle="--", label="Reference value")]
fig.legend(handles=handles2, loc="lower center", ncol=min(len(EXP) + 1, 8), frameon=False,
           bbox_to_anchor=(0.5, -0.05), fontsize=9)
fig.suptitle("Twin experiments — best-individual parameter values (SBX, pop 256)", fontweight="bold", fontsize=11)
plt.tight_layout(rect=[0, 0.04, 1, 0.96])
fig.savefig(FIG / "Figure_8.png", dpi=200, bbox_inches="tight", facecolor="white")
print("wrote", FIG / "Figure_8.png")

# ---------------- Table 1 optimised columns ----------------
rows = [{"experiment": "REFERENCE", **{p: REF[p] for p in PARAM_ORDER}}]
for e in EXP:
    rows.append({"experiment": DISPLAY.get(e, e), **{p: best[e][p] for p in PARAM_ORDER}})
t1 = pd.DataFrame(rows).set_index("experiment")
t1.to_csv(LOGS / "table1_optimised.csv")
print("\n===== Table 1 (optimised parameters) =====")
print(t1.to_string(float_format=lambda x: f"{x:.4g}"))
print("\nwrote", LOGS / "table1_optimised.csv")
