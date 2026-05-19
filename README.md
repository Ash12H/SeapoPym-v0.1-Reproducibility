# SeapoPym v0.1 — Reproducibility deposit

Code, configuration and pipeline accompanying:

> Lehodey, J.V., Mignot, A., Ganachaud, A., Albernhe, S., Nicol, S. (2026).
> *SeapoPym v0.1: Implementation of the SEAPODYM low and mid trophic levels in
> Python with a flexible optimisation framework*.
> Geoscientific Model Development (in review). DOI: 10.5194/egusphere-2026-711.

## What this deposit contains

A self-contained chain that reproduces every figure and table of the manuscript.
Step-by-step instructions to set up the environment, download the forcings and
run the notebooks are in [`docs/REPRODUCTION.md`](docs/REPRODUCTION.md).

## Figure-to-notebook map

| Output | Notebook |
|---|---|
| Figure 1 (global forcing maps) | `notebooks/02_global_simulation/01_forcing_maps.ipynb` |
| Figure 2 + Table 2 (stations in T–NPP space) | `notebooks/04_twin_experiments/stations_distribution.ipynb` |
| Figure 3 (theoretical benchmark) | `notebooks/01_theoretical_validation/theoretical_benchmark.ipynb` |
| Figure 4 (transport impact: SeapoPym vs LMTL) | `notebooks/02_global_simulation/03_transport_impact.ipynb` |
| Figure 5 (Sobol sensitivity indices) | `notebooks/03_sobol_sensitivity/03_analyze_sobol_indices.ipynb` |
| Figure 6 (twin experiment NRMSE) | `notebooks/04_twin_experiments/03_nrmse_evolution.ipynb` |
| Figure 7 (twin experiment MAPE) | `notebooks/04_twin_experiments/04_mape_evolution.ipynb` |
| Figure 8 (twin experiment HDI) | `notebooks/04_twin_experiments/05_hdi_evolution.ipynb` |

## Layout

```
seapopym-v0.1-reproducibility/
├── README.md                  # this file
├── LICENSE                    # GPL-3.0-or-later
├── CITATION.cff               # citation metadata
├── pyproject.toml + uv.lock   # pinned uv-managed Python environment
├── parameters.yaml            # parameter values, bounds, and run modes
├── data/
│   └── stations_coords.json   # 6 oceanographic stations (Table 2)
├── notebooks/
│   ├── 01_theoretical_validation/
│   ├── 02_global_simulation/
│   ├── 03_sobol_sensitivity/
│   └── 04_twin_experiments/
├── scripts/
│   ├── download_cmems_stations.py
│   ├── download_cmems_global.py
│   └── run_optimization.py
├── figures/                   # output figures (regenerated from the notebooks)
├── figures_published/         # copies of the manuscript figures (for comparison)
└── docs/
    └── REPRODUCTION.md        # full setup + reproduction guide
```

The Zarr archives and Parquet logbooks are not redistributed in this deposit; the
scripts and notebooks regenerate them from public sources (see `docs/REPRODUCTION.md`).

## License

GPL-3.0-or-later (see [`LICENSE`](LICENSE)).
