# Setup and reproduction

This document covers everything needed to install the environment, fetch the
forcings, and reproduce the figures of the manuscript on a workstation.

## 1. Prerequisites

- A POSIX-like shell (macOS or Linux; Windows users should use WSL).
- [`uv`](https://docs.astral.sh/uv/) — the Python environment manager:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
- A free [Copernicus Marine](https://marine.copernicus.eu) account.
- About 7 GB of free disk space if you reproduce all eight figures (under 1 GB
  if you skip Figures 1, 2, 4).

## 2. Install the environment

From the root of this deposit:

```bash
uv sync
```

This creates `.venv/` and installs every dependency at the version pinned in
`uv.lock`, including the SeapoPym source code at the manuscript revision
(tag `v0.1`).

## 3. Authenticate against Copernicus Marine

One-off interactive step:

```bash
uv run copernicusmarine login
```

The credentials are stored in `~/.copernicusmarine/` and reused automatically
by the download scripts.

## 4. Fetch the forcings

Two scripts are provided. Run the one(s) you need depending on which figures
you intend to reproduce.

### Six stations only (Figures 3, 5, 6, 7, 8)

```bash
uv run python scripts/download_cmems_stations.py
```

Output: `data/stations.zarr` (~500 KB). Runtime: about 30 seconds to a few
minutes on a residential link.

### Full global field (additionally required for Figures 1, 2, 4)

```bash
uv run python scripts/download_cmems_global.py
```

Output: `data/forcings_global.zarr` (~6 GB). Runtime: typically a few hours
on a residential link, bandwidth-bound. The script processes the data
month by month and is resumable — re-run the same command after an
interruption and it will pick up where it left off.

Both scripts share the same 1° bin-averaging algorithm; the values they
produce at the six station coordinates are bit-for-bit identical.

The source dataset is the Copernicus Marine product
**GLOBAL_MULTIYEAR_BGC_001_033** (DOI [10.48670/moi-00020](https://doi.org/10.48670/moi-00020)),
1998-01-01 to 2019-12-31, daily, regridded from native 1/12° to 1°.

## 5. Run the notebooks

The default run mode is `test`, set in `parameters.yaml`. Test mode runs the
heavy pipelines (Sobol, genetic algorithm) end-to-end in about 12 minutes on
a recent workstation; the qualitative structure of every figure matches the
published version, but the Sobol indices have wider confidence intervals and
the GA trajectories show less convergence than in `production` mode.

Switch by editing `parameters.yaml` and setting `mode: production` in the
sections you want. Production mode reproduces the manuscript values exactly;
the GA in particular takes several hours to days on a workstation and was
performed on a high-performance cluster for the manuscript.

Recommended execution order:

```
notebooks/01_theoretical_validation/theoretical_benchmark.ipynb     # Figure 3

notebooks/02_global_simulation/                                     # needs forcings_global.zarr
    01_forcing_maps.ipynb                                           # Figure 1
    02_run_global_simulation.ipynb                                  # produces data/biomass_global.zarr
    03_transport_impact.ipynb                                       # Figure 4

notebooks/04_twin_experiments/                                      # needs stations.zarr
    stations_distribution.ipynb                                     # Figure 2 + Table 2
                                                                    #   (also needs forcings_global.zarr
                                                                    #    for the background density)
    01_generate_pseudo_observations.ipynb                           # produces pseudo_observations.zarr
                                                                    #   + initial_conditions.zarr
    02_run_all_experiments.ipynb                                    # runs the 7 GA experiments
    03_nrmse_evolution.ipynb                                        # Figure 6
    04_mape_evolution.ipynb                                         # Figure 7
    05_hdi_evolution.ipynb                                          # Figure 8

notebooks/03_sobol_sensitivity/                                     # needs stations.zarr
    01_generate_sobol_samples.ipynb
    02_run_sobol_simulations.ipynb
    03_analyze_sobol_indices.ipynb                                  # Figure 5
```

Either open `jupyter lab` and run each notebook interactively, or batch all of
them:

```bash
for nb in \
    notebooks/01_theoretical_validation/theoretical_benchmark.ipynb \
    notebooks/02_global_simulation/01_forcing_maps.ipynb \
    notebooks/02_global_simulation/02_run_global_simulation.ipynb \
    notebooks/02_global_simulation/03_transport_impact.ipynb \
    notebooks/04_twin_experiments/01_generate_pseudo_observations.ipynb \
    notebooks/04_twin_experiments/stations_distribution.ipynb \
    notebooks/04_twin_experiments/02_run_all_experiments.ipynb \
    notebooks/04_twin_experiments/03_nrmse_evolution.ipynb \
    notebooks/04_twin_experiments/04_mape_evolution.ipynb \
    notebooks/04_twin_experiments/05_hdi_evolution.ipynb \
    notebooks/03_sobol_sensitivity/01_generate_sobol_samples.ipynb \
    notebooks/03_sobol_sensitivity/02_run_sobol_simulations.ipynb \
    notebooks/03_sobol_sensitivity/03_analyze_sobol_indices.ipynb
do
    uv run jupyter nbconvert --to notebook --execute "$nb" --output "$(basename "$nb")"
done
```

Generated figures land in `figures/` as PDF and PNG; the published versions sit
in `figures_published/` for direct comparison.

## Runtime expectations

| Stage                                  | Test mode              | Production mode              |
| -------------------------------------- | ---------------------- | ---------------------------- |
| `uv sync` (one-off)                    | ~1 min                 | ~1 min                       |
| `download_cmems_stations.py` (one-off) | ~30 s to a few minutes | same                         |
| `download_cmems_global.py` (one-off)   | a few hours            | same                         |
| Figure 1 + Figure 4 (global)           | ~1 min                 | ~5 min                       |
| Figure 2 + Table 2                     | ~30 s                  | ~30 s                        |
| Figure 3                               | ~5 s                   | ~5 s                         |
| Figure 5 (Sobol)                       | ~5 min                 | several hours                |
| Figures 6, 7, 8 (GA × 7 stations)      | ~6 min total           | several hours                |
| **End-to-end after one-off downloads** | **~12-16 min**         | multi-hour, cluster-friendly |

## What is not in the deposit

- The CMEMS forcings themselves (~6 GB after regridding). They are publicly
  available from Copernicus Marine and fetched on demand by the scripts in
  `scripts/`.
- The genetic-algorithm logbooks used to produce the published Figures 6, 7, 8.
  These were generated on a high-performance cluster (~1.4 million model
  evaluations); they are regenerable with `scripts/run_optimization.py` in
  production mode.
