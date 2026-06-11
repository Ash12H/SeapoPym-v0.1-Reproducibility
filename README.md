# SeapoPym v0.1 — reproducibility deposit

Code and data accompanying Lehodey et al. (2026, *Geoscientific Model Development*):
**SeapoPym v0.1** (egusphere-2026-711).

> ⚠️ **Work in progress.** The deposit is being restructured for the revision. Full
> setup/reproduction instructions and the figure-to-script provenance map will be added once
> the revised analysis is finalised.

## Layout

```
src/seapopym_repro/   shared package (paths, twin-experiment core, figure style)
scripts/
  data/               forcing download (CMEMS) + pseudo-observation generation
  experiments/        CMA-ES calibration ensemble + Sobol sensitivity
  figures/            figure scripts (read products/, write figures/)
data/                 inputs — small station forcings tracked; heavy global forcing fetched via scripts/data/
products/             frozen experiment outputs (CSV) the figures consume
figures/              produced figures (PDF + PNG)
parameters.yaml       reference parameters, bounds, run configuration
```

Calibration uses **CMA-ES** (pycma); the environment is pinned with `uv` (`pyproject.toml` + `uv.lock`).
