# Published figures (manuscript reference)

Exact copies of the 8 figures published in the manuscript
(Lehodey et al. 2026, *Geoscientific Model Development*, in review).

Provided here so that anyone reproducing the pipeline can compare the
regenerated figures (in `../figures/`) against the originals without
opening the PDF of the paper.

## Contents

| File | Manuscript figure | Produced by |
|---|---|---|
| `Figure_1.{pdf,png}` | Figure 1 — global forcing maps | `notebooks/02_global_simulation/01_forcing_maps.ipynb` |
| `Figure_2.{pdf,png}` | Figure 2 — stations in T-NPP space | `notebooks/04_twin_experiments/stations_distribution.ipynb` |
| `Figure_3.{pdf,png}` | Figure 3 — theoretical asymptote benchmark | `notebooks/01_theoretical_validation/theoretical_benchmark.ipynb` |
| `Figure_4a.{pdf,png}` | Figure 4a — SeapoPym mean biomass | `notebooks/02_global_simulation/03_transport_impact.ipynb` |
| `Figure_4b.{pdf,png}` | Figure 4b — Spatial RMSE vs LMTL | same notebook |
| `Figure_5.{pdf,png}` | Figure 5 — Sobol indices (S1, ST) | `notebooks/03_sobol_sensitivity/03_analyze_sobol_indices.ipynb` |
| `Figure_6.{pdf,png}` | Figure 6 — NRMSE evolution | `notebooks/04_twin_experiments/03_nrmse_evolution.ipynb` |
| `Figure_7.{pdf,png}` | Figure 7 — MAPE evolution per parameter | `notebooks/04_twin_experiments/04_mape_evolution.ipynb` |
| `Figure_8.{pdf,png}` | Figure 8 — HDI evolution | `notebooks/04_twin_experiments/05_hdi_evolution.ipynb` |

Note: in this deposit Figure 4 is generated as a single combined notebook
output (`figures/Figure_4.{pdf,png}`), whereas the manuscript displays the
two panels (4a, 4b) as separate sub-figures.

## How to compare

After running the pipeline (see `../docs/REPRODUCTION.md`):

```bash
# side-by-side visual check on macOS:
open figures_published/Figure_5.png figures/Figure_5.png
```

In `test` mode, Figures 5 and 6-8 will have wider confidence intervals
and noisier trajectories than the published versions because of the
reduced sample sizes; the qualitative structure should match.

In `production` mode, the generated figures should be visually
indistinguishable from those in `figures_published/`.
