"""run_sobol_production.py — REHEARSAL of the final Sobol sensitivity run.

Isolated dress rehearsal (mirrors rehearsal/run_ga_production.py), separate from the
notebooks in notebooks/03_sobol_sensitivity/ and from data/sobol_*.parquet (the
published explicit-solver results). Validates the unified-config Sobol pipeline BEFORE
touching any official notebook.

What changed vs the published run:
  - SOLVER     : implicit (semi-implicit IMEX) biomass scheme — unconditionally stable,
                 so the warm-station high-mortality corner no longer blows up (no NaN).
  - BOUNDS     : the *unified* GA bounds (parameters.yaml -> model_parameters.bounds),
                 NOT the wider sobol.bounds. One bounds set for the whole study.
  - MODEL      : NoTransportSpaceOptimizedLightModel (same class as the GA; faster/lighter
                 for 1.19 M sims; biomass identical to the full model).
  - SPIN-UP    : 2-year spin-up baked into the analysis window (parameters.yaml:
                 sobol.analysis_period.spin_up_start 2004 -> start 2006).

The runner stores only the raw per-sample (mean, variance, argmax) per station. The robust
metrics (log10 mean, CV, circular timing) that REPLACE them in Figure 5 are derived
downstream by the analysis script (experiment/05_sobol_robust_metrics.py) from these.

RESUMABLE: results are written batch-by-batch to rehearsal/sobol/sobol_results.parquet.
Re-running after an interruption (Ctrl-C, crash, shutdown) reloads what is already there and
skips completed batches — exactly the mechanism from notebook 02.

Run:
    .venv/bin/python rehearsal/run_sobol_production.py                 # uses sobol.mode in yaml
    .venv/bin/python rehearsal/run_sobol_production.py --mode test     # 12,288 sims (~8 min)
    .venv/bin/python rehearsal/run_sobol_production.py --mode production
    .venv/bin/python rehearsal/run_sobol_production.py --mode test --limit 200   # quick smoke
    .venv/bin/python rehearsal/run_sobol_production.py --workers 8 --batch-size 2000
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml
from dask.distributed import Client
from SALib import ProblemSpec
from seapopym.configuration.no_transport import (
    ForcingParameter,
    ForcingUnit,
    FunctionalGroupParameter,
    FunctionalGroupUnit,
    FunctionalTypeParameter,
    KernelParameter,
    MigratoryTypeParameter,
    NoTransportConfiguration,
)
from seapopym.model.no_transport_model import NoTransportSpaceOptimizedLightModel

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "rehearsal" / "sobol"

SOLVER = "implicit"
SEED = 0  # reproducible Saltelli sample


def build_forcing_and_grid(period):
    """Reshape the 6 station time series into a compact (T, Y, X) grid (as in notebook 02)."""
    raw = xr.open_zarr(DATA_DIR / "stations.zarr").sel(
        time=slice(period["spin_up_start"], period["end"])
    ).load()
    ordered = raw.isel(station=raw.station_lat.argsort().values)
    y_values = ordered.station_lat.values.astype(np.float32)
    x_values = np.array([0.0], dtype=np.float32)

    def to_grid(da):
        arr = da.values.T[:, :, np.newaxis]  # (station, time) -> (T, Y, X)
        return xr.DataArray(arr, dims=("T", "Y", "X"),
                            coords={"T": ordered.time.values, "Y": ("Y", y_values), "X": ("X", x_values)})

    temperature = to_grid(ordered.temperature).expand_dims(Z=[0], axis=1)
    npp = to_grid(ordered.npp)
    for coord, axis in {"T": "T", "Z": "Z", "Y": "Y", "X": "X"}.items():
        if coord in temperature.coords:
            temperature[coord].attrs["axis"] = axis
        if coord in npp.coords:
            npp[coord].attrs["axis"] = axis
    temperature.attrs["units"] = "degC"
    npp.attrs["units"] = "mg/m2/day"

    forcing = ForcingParameter(
        temperature=ForcingUnit(forcing=temperature),
        primary_production=ForcingUnit(forcing=npp),
    )
    station_grid = pd.DataFrame(
        {"Y": y_values.astype(float), "X": np.zeros(len(y_values))},
        index=ordered.station.values,
    )
    return forcing, station_grid


def evaluate_sample(params, forcing_parameter, station_grid, t_analysis):
    """One SeapoPym run (implicit solver); return [mean, var, argmax] per station, flattened."""
    fg = FunctionalGroupParameter(functional_group=[FunctionalGroupUnit(
        name="zooplankton",
        energy_transfert=params[0],
        functional_type=FunctionalTypeParameter(
            lambda_temperature_0=params[3],
            gamma_lambda_temperature=params[4],
            tr_0=params[1],
            gamma_tr=params[2],
        ),
        migratory_type=MigratoryTypeParameter(day_layer=0, night_layer=0),
    )])
    config = NoTransportConfiguration(
        forcing=forcing_parameter,
        functional_group=fg,
        kernel=KernelParameter(biomass_solver=SOLVER),
    )
    with NoTransportSpaceOptimizedLightModel.from_configuration(config) as model:
        model.run()
        out = []
        for _, row in station_grid.iterrows():
            b = model.state.biomass.sel(T=t_analysis, Y=row["Y"], X=row["X"], functional_group=0)
            try:
                argmax_val = int(b.argmax("T"))
            except ValueError:
                argmax_val = -1
            out.append([float(b.mean()), float(b.var()), argmax_val])
    return np.array(out).flatten()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["test", "production"], default=None,
                    help="override sobol.mode from parameters.yaml")
    ap.add_argument("--n", type=int, default=None,
                    help="override n_samples_per_dim (base N); total = N*(2D+2) or N*(D+2)")
    ap.add_argument("--out-subdir", default=None,
                    help="write outputs under rehearsal/sobol/<subdir>/ (e.g. a convergence point)")
    ap.add_argument("--second-order", action="store_true",
                    help="compute pairwise S2 (design 2D+2). Default: first-order design D+2 (no S2).")
    ap.add_argument("--limit", type=int, default=None, help="evaluate only the first N samples (smoke test)")
    ap.add_argument("--batch-size", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--memory-limit", default="4GB")
    args = ap.parse_args()

    out_dir = OUT_DIR if not args.out_subdir else OUT_DIR / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(ROOT / "parameters.yaml") as f:
        params = yaml.safe_load(f)
    sobol = params["sobol"]
    mode = args.mode or sobol["mode"]
    names = sobol["parameters_ordered"]
    metrics = sobol["metrics"]
    period = sobol["analysis_period"]
    n_samples = args.n if args.n is not None else sobol[mode]["n_samples_per_dim"]
    calc_second_order = args.second_order  # default False: only S1/ST are reported (Figure 6)

    # UNIFIED bounds = the GA bounds (not sobol.bounds).
    bounds = [params["model_parameters"]["bounds"][p] for p in names]

    samples_path = out_dir / "sobol_samples.parquet"
    results_path = out_dir / "sobol_results.parquet"
    meta_path = out_dir / "run_meta.json"

    # ---- 1. Saltelli sample from the unified GA bounds (regenerated; reproducible seed) ----
    factor = (2 * len(names) + 2) if calc_second_order else (len(names) + 2)
    expected_total = n_samples * factor
    if samples_path.exists() and len(pd.read_parquet(samples_path)) == expected_total:
        samples = pd.read_parquet(samples_path)
        print(f"Reusing existing sample: {len(samples):,} rows ({samples_path.name})", flush=True)
    else:
        sp = ProblemSpec({"names": names, "groups": None, "bounds": bounds, "outputs": metrics})
        sp.sample_sobol(n_samples, calc_second_order=calc_second_order, seed=SEED)
        samples = pd.DataFrame(sp.samples, columns=names)
        samples.to_parquet(samples_path)
        print(f"Generated Saltelli sample: {len(samples):,} rows (N={n_samples:,}, "
              f"{'2nd' if calc_second_order else '1st'}-order design) -> {samples_path.name}", flush=True)

    if args.limit is not None:
        samples = samples.iloc[: args.limit]
        print(f"--limit {args.limit}: evaluating only the first {len(samples):,} samples", flush=True)

    meta_path.write_text(json.dumps({
        "solver": SOLVER, "mode": mode, "bounds_source": "model_parameters.bounds (unified GA bounds)",
        "names": names, "bounds": {n: b for n, b in zip(names, bounds)}, "metrics": metrics,
        "n_samples_per_dim": n_samples, "calc_second_order": calc_second_order,
        "total_samples": int(expected_total),
        "analysis_period": period, "model": "NoTransportSpaceOptimizedLightModel",
    }, indent=2))

    # ---- 2. Resumable batch evaluation (implicit solver) ----
    forcing, station_grid = build_forcing_and_grid(period)
    columns = pd.MultiIndex.from_product([station_grid.index, metrics], names=["station", "metric"])
    t_analysis = slice(period["start"], period["end"])

    if results_path.exists():
        results = pd.read_parquet(results_path)
        print(f"Resuming: {len(results):,} / {len(samples):,} already computed.", flush=True)
    else:
        results = None

    client = Client(n_workers=args.workers, threads_per_worker=1, memory_limit=args.memory_limit)
    print(f"Dask dashboard: {client.dashboard_link}", flush=True)
    print(f"mode={mode} | solver={SOLVER} | samples={len(samples):,} | batch={args.batch_size} "
          f"| bounds=GA", flush=True)
    forcing_future = client.scatter(forcing, broadcast=True)

    batch_size = args.batch_size
    n_batches = (len(samples) + batch_size - 1) // batch_size
    t0 = time.time()
    try:
        for b in range(n_batches):
            lo, hi = b * batch_size, min((b + 1) * batch_size, len(samples))
            if results is not None and (hi - 1) in results.index:
                continue
            batch = samples.iloc[lo:hi]
            futures = client.map(
                evaluate_sample, batch.to_numpy(),
                forcing_parameter=forcing_future, station_grid=station_grid, t_analysis=t_analysis,
            )
            out = pd.DataFrame(client.gather(futures), columns=columns, index=batch.index).astype(float)
            results = out if results is None else pd.concat([results, out])
            results.to_parquet(results_path)
            print(f"  batch {b+1}/{n_batches}: rows {lo}-{hi-1} "
                  f"({len(results):,}/{len(samples):,}, {(time.time()-t0)/60:.1f} min)", flush=True)
    finally:
        client.close()

    # ---- 3. Stability diagnostic: with the implicit solver everything must be finite ----
    means = results.xs("mean", axis=1, level="metric").to_numpy(dtype=float)
    varis = results.xs("variance", axis=1, level="metric").to_numpy(dtype=float)
    argmaxs = results.xs("argmax", axis=1, level="metric").to_numpy(dtype=float)
    n_bad_mean = int((~np.isfinite(means)).sum())
    n_bad_var = int((~np.isfinite(varis)).sum())
    n_bad_argmax = int((argmaxs == -1).sum())
    print("\n===== STABILITY DIAGNOSTIC (implicit solver) =====", flush=True)
    print(f"  rows evaluated      : {len(results):,}", flush=True)
    print(f"  non-finite mean     : {n_bad_mean}", flush=True)
    print(f"  non-finite variance : {n_bad_var}", flush=True)
    print(f"  argmax == -1 (NaN b): {n_bad_argmax}", flush=True)
    clean = (n_bad_mean == 0 and n_bad_var == 0 and n_bad_argmax == 0)
    print(f"  -> {'CLEAN: 0 NaN across the full GA-bounds box' if clean else 'WARNING: non-finite outputs present'}",
          flush=True)
    print(f"\nResults: {results_path}\nMeta:    {meta_path}", flush=True)


if __name__ == "__main__":
    main()
