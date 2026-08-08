"""Run the global SeapoPym simulation behind Figure 4.

Runs SeapoPym at 1° resolution with the reference parameters of Table 1, on the temperature
and NPP forcings of data/forcings_global.zarr, over the full forcing window. The first two years
are a spin-up and are discarded: only the analysis period (2000-01-01 .. end_date) is written.

This is the transport-free side of the comparison against the SEAPODYM-LMTL operational reference
(the `zooc` field of forcings_global.zarr), which freeze_transport_maps.py reduces to the maps of
Figure 4.

The model runs as a single zooplankton functional group and computes its own initial state. It uses
the explicit biomass solver, whose equilibrium is the same as the implicit one; only the approach
time differs, and the two-year spin-up absorbs that difference. Reference parameters and dates come
from parameters.yaml.

Inputs : data/forcings_global.zarr (temperature, npp), parameters.yaml
Output : data/biomass_global.zarr ('biomass', analysis period only)
Run    : .venv/bin/python scripts/data/run_global_simulation.py
Pin BLAS threads first: export OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
                               MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta

import xarray as xr
from dask.distributed import Client
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
from seapopym.model.no_transport_model import NoTransportModel

from seapopym_repro import paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--forcing-end",
        default=None,
        help="Override the forcing/analysis end date (YYYY-MM-DD). Defaults to "
        "global_simulation.end_date. Use a short value (e.g. 1998-12-31) for a quick timing test.",
    )
    parser.add_argument(
        "--timing",
        action="store_true",
        help="Time the run without writing biomass_global.zarr.",
    )
    args = parser.parse_args()

    params = paths.load_params()
    ref = params["model_parameters"]["reference"]
    gs = params["global_simulation"]

    t_start = gs["start_date"]
    t_end = args.forcing_end or gs["end_date"]
    t_analysis_start = (
        datetime.strptime(gs["spin_up_end"], "%Y-%m-%d") + timedelta(days=1)
    ).strftime("%Y-%m-%d")
    print(
        f"forcing: {t_start} -> {t_end} | spin-up ends {gs['spin_up_end']} | "
        f"analysis output: {t_analysis_start} -> {t_end}",
        flush=True,
    )

    client = Client(n_workers=2, threads_per_worker=1, memory_limit="16GB")
    print(client.dashboard_link, flush=True)

    forcings = xr.open_zarr(paths.DATA / "forcings_global.zarr").sel(T=slice(t_start, t_end)).load()
    print(f"loaded forcings: {dict(forcings.sizes)}", flush=True)

    # SeapoPym expects a (T, Z, Y, X) shape for temperature; Z=0 matches day/night_layer below.
    temperature = forcings.temperature.expand_dims(Z=[0], axis=1)
    for coord, axis in {"T": "T", "Z": "Z", "Y": "Y", "X": "X"}.items():
        if coord in temperature.coords:
            temperature[coord].attrs["axis"] = axis
        if coord in forcings.npp.coords:
            forcings.npp[coord].attrs["axis"] = axis

    functional_group = FunctionalGroupParameter(
        functional_group=[
            FunctionalGroupUnit(
                name="zooplankton",
                energy_transfert=ref["energy_transfert"],
                functional_type=FunctionalTypeParameter(
                    lambda_temperature_0=ref["lambda_temperature_0"],
                    gamma_lambda_temperature=ref["gamma_lambda_temperature"],
                    tr_0=ref["tr_0"],
                    gamma_tr=ref["gamma_tr"],
                ),
                migratory_type=MigratoryTypeParameter(day_layer=0, night_layer=0),
            )
        ]
    )

    config = NoTransportConfiguration(
        forcing=ForcingParameter(
            temperature=ForcingUnit(forcing=temperature),
            primary_production=ForcingUnit(forcing=forcings.npp),
        ),
        functional_group=functional_group,
        kernel=KernelParameter(compute_initial_conditions=True),
    )

    import time as _time

    _t0 = _time.time()
    with NoTransportModel.from_configuration(configuration=config) as model:
        model.run()
        model.state.compute()
        biomass = model.state.biomass.load().copy()
    print(
        f"model.run + compute took {_time.time() - _t0:.1f} s for nT={biomass.sizes['T']}",
        flush=True,
    )

    if args.timing:
        client.close()
        n_full = 8035
        per = (_time.time() - _t0) / biomass.sizes["T"]
        print(f"Timing only, nothing saved. Full 22 years extrapolates to ~{per * n_full / 60:.1f} min", flush=True)
        return

    output_path = paths.DATA / "biomass_global.zarr"
    biomass.sel(T=slice(t_analysis_start, t_end)).to_zarr(output_path, mode="w", zarr_format=2)
    client.close()
    size_mb = sum(f.stat().st_size for f in output_path.rglob("*") if f.is_file()) / 1e6
    print(f"Wrote {output_path.name} | size = {size_mb:.1f} MB", flush=True)


if __name__ == "__main__":
    main()
