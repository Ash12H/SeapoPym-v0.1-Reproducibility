"""Evaluate one Sobol sample of the five biological parameters at the six stations.

Draws a Saltelli sample of base size N over the parameter bounds of parameters.yaml, runs SeapoPym
once per parameter set at each station, and records three descriptors of the resulting biomass
series: its mean, its variance and the day of year of its maximum. freeze_sobol_indices.py then
turns these into Sobol indices.

The runs use the implicit biomass solver, which stays stable over the whole parameter range: the
explicit scheme diverges in the high-mortality corner at the warm stations. The analysis window
starts two years after the beginning of the forcing, so every run is scored on a spun-up state.

Results are written batch by batch, so an interrupted run resumes and skips the batches already
computed.

This script is normally driven by run_sobol_convergence.py, which calls it once per sample size.

Inputs : data/stations.zarr, parameters.yaml
Output : results_raw/sobol/<subdir>/sobol_results.parquet, run_meta.json
Run    : .venv/bin/python scripts/experiments/run_sobol_production.py
         .venv/bin/python scripts/experiments/run_sobol_production.py --n 8192
         .venv/bin/python scripts/experiments/run_sobol_production.py --workers 8 --batch-size 2000
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

from seapopym_repro import paths
ROOT = paths.ROOT
DATA_DIR = paths.DATA
OUT_DIR = paths.RESULTS_RAW / "sobol"

SOLVER = "implicit"
SEED = 0  # reproducible Saltelli sample


def build_forcing_and_grid(period):
    """Reshape the six station time series into a compact (T, Y, X) grid."""
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
    ap.add_argument("--n", type=int, default=None,
                    help="override n_samples_per_dim (base N); total = N*(2D+2) or N*(D+2)")
    ap.add_argument("--out-subdir", default=None,
                    help="write outputs under results_raw/sobol/<subdir>/, one per convergence point")
    ap.add_argument("--second-order", action="store_true",
                    help="compute pairwise S2 indices, a 2D+2 design; the default D+2 design omits them")
    ap.add_argument("--limit", type=int, default=None, help="evaluate only the first N samples, to test the setup")
    ap.add_argument("--batch-size", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--memory-limit", default="4GB")
    args = ap.parse_args()

    out_dir = OUT_DIR if not args.out_subdir else OUT_DIR / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(ROOT / "parameters.yaml") as f:
        params = yaml.safe_load(f)
    sobol = params["sobol"]
    names = sobol["parameters_ordered"]
    metrics = sobol["metrics"]
    period = sobol["analysis_period"]
    n_samples = args.n if args.n is not None else sobol["n_samples_per_dim"]
    calc_second_order = args.second_order  # default False: only S1/ST are reported (Figure 6)

    # One set of bounds for the whole study, the one the twin experiments search over.
    bounds = [params["model_parameters"]["bounds"][p] for p in names]

    samples_path = out_dir / "sobol_samples.parquet"
    results_path = out_dir / "sobol_results.parquet"
    meta_path = out_dir / "run_meta.json"

    # ---- 1. Saltelli sample over those bounds, from a fixed seed so the design is reproducible ----
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
        "solver": SOLVER, "mode": "production", "bounds_source": "model_parameters.bounds",
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
    print(f"solver={SOLVER} | samples={len(samples):,} | batch={args.batch_size} | bounds=GA", flush=True)
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
