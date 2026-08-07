"""Theoretical benchmark experiment — freeze the biomass-convergence curves for Figure 3.

Runs the SeapoPym model under CONSTANT forcing at four temperatures (0/10/20/30 C, NPP = 300
mg C m-2 d-1) and records its convergence toward the analytical steady state B = R / lambda(T),
with recruitment R = E * NPP. The result is a small committed product so Figure 3 redraws from git
WITHOUT the framework (only this CSV is needed at figure time).

Output : products/benchmark_convergence.csv   (temperature, day, biomass, asymptote)
Run    : .venv/bin/python scripts/experiments/run_benchmark.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
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

PARAMS = paths.load_params()
REF = PARAMS["model_parameters"]["reference"]
BENCH = PARAMS["theoretical_benchmark"]
# The 0 C case is the slowest: the biomass relaxes toward B = R / lambda with time constant
# 1 / lambda_0 = 150 d, so reaching the analytical steady state to <0.01% takes ~9.2 / lambda_0
# ~ 1400 d, and to numerical precision ~16 / lambda_0 ~ 2400 d. Run seven years so every
# temperature (0 C included) actually converges within the window.
DURATION_DAYS = 2555


def transform_temperature(temperature: float) -> float:
    return temperature / (1 + temperature / 273)


def mortality_rate(temperature: float) -> float:
    return REF["lambda_temperature_0"] * np.exp(REF["gamma_lambda_temperature"] * temperature)


def analytical_asymptote(t_celsius: float) -> float:
    """Steady state B = R / lambda(T), with recruitment R = E * NPP."""
    return REF["energy_transfert"] * BENCH["npp_constant"] / mortality_rate(transform_temperature(t_celsius))


def build_constant_forcing(t_celsius: float, duration_days: int = DURATION_DAYS):
    time = pd.date_range("2000-01-01", periods=duration_days, freq="D")
    coord_attrs = {"time": {"axis": "T"}, "latitude": {"axis": "Y"},
                   "longitude": {"axis": "X"}, "depth": {"axis": "Z"}}
    temperature = xr.DataArray(
        np.full((duration_days, 1, 1, 1), t_celsius, dtype=np.float32),
        dims=("time", "depth", "latitude", "longitude"),
        coords={"time": time, "depth": [1], "latitude": [0.5], "longitude": [0.5]},
        attrs={"units": "degC"},
    )
    npp = xr.DataArray(
        np.full((duration_days, 1, 1), BENCH["npp_constant"], dtype=np.float32),
        dims=("time", "latitude", "longitude"),
        coords={"time": time, "latitude": [0.5], "longitude": [0.5]},
        attrs={"units": "mg/m2/day"},
    )
    for da in (temperature, npp):
        for coord_name, attrs in coord_attrs.items():
            if coord_name in da.coords:
                da[coord_name].attrs.update(attrs)
    return temperature, npp


def run_benchmark(t_celsius: float) -> xr.DataArray:
    temperature, npp = build_constant_forcing(t_celsius)
    functional_group = FunctionalGroupParameter(functional_group=[FunctionalGroupUnit(
        name=f"{t_celsius}deg",
        energy_transfert=REF["energy_transfert"],
        functional_type=FunctionalTypeParameter(
            lambda_temperature_0=REF["lambda_temperature_0"],
            gamma_lambda_temperature=REF["gamma_lambda_temperature"],
            tr_0=REF["tr_0"], gamma_tr=REF["gamma_tr"],
        ),
        migratory_type=MigratoryTypeParameter(day_layer=1, night_layer=1),
    )])
    config = NoTransportConfiguration(
        forcing=ForcingParameter(
            temperature=ForcingUnit(forcing=temperature),
            primary_production=ForcingUnit(forcing=npp),
        ),
        functional_group=functional_group,
        # Implicit biomass scheme, the one the paper describes and the one every other experiment
        # uses. Both schemes share the same analytical equilibrium B = R / lambda, so this changes
        # only the approach time, and only in warm water where lambda * dt is no longer small.
        kernel=KernelParameter(
            compute_initial_conditions=False, compute_preproduction=False, biomass_solver="implicit"
        ),
    )
    with NoTransportModel.from_configuration(configuration=config) as model:
        model.run()
        biomass = model.state.biomass.copy()
    return biomass.pint.quantify().pint.to("mg/m2").pint.dequantify()


def main():
    rows = []
    for t in BENCH["temperatures_celsius"]:
        series = run_benchmark(t).isel(X=0, Y=0).squeeze().values
        asym = analytical_asymptote(t)
        for day, b in enumerate(series, start=1):   # 1-based elapsed days (log x-axis)
            rows.append({"temperature": t, "day": day, "biomass": float(b), "asymptote": float(asym)})
    paths.PRODUCTS.mkdir(parents=True, exist_ok=True)
    out = paths.PRODUCTS / "benchmark_convergence.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"froze benchmark -> {out}  ({len(rows)} rows, {len(BENCH['temperatures_celsius'])} temperatures)")


if __name__ == "__main__":
    main()
