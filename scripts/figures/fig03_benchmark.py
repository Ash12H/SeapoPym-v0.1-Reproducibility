"""Figure 3 — theoretical benchmark: convergence to the analytical asymptote.

Numerical convergence of the SeapoPym biomass to B_eq = E * NPP / lambda(T) under
constant forcing, at four temperatures (0, 10, 20, 30 °C), NPP = 300 mg C m-2 d-1.

Legend (RC3 #50 fix): line COLOUR = temperature; DASHED line = analytical asymptote.
X axis in elapsed days on a LOG scale (revision request).

Output: rehearsal/figures/Figure_3.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import yaml
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
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

from seapopym_repro import figstyle as fs, paths
ROOT = paths.ROOT

with open(ROOT / "parameters.yaml") as f:
    PARAMS = yaml.safe_load(f)
REF = PARAMS["model_parameters"]["reference"]
BENCH = PARAMS["theoretical_benchmark"]
DURATION_DAYS = 540


def transform_temperature(temperature: float) -> float:
    return temperature / (1 + temperature / 273)


def mortality_rate(temperature: float) -> float:
    return REF["lambda_temperature_0"] * np.exp(REF["gamma_lambda_temperature"] * temperature)


def analytical_asymptote(t_celsius: float) -> float:
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
        kernel=KernelParameter(compute_initial_conditions=False, compute_preproduction=False),
    )
    with NoTransportModel.from_configuration(configuration=config) as model:
        model.run()
        biomass = model.state.biomass.copy()
    return biomass.pint.quantify().pint.to("mg/m2").pint.dequantify()


temperatures = BENCH["temperatures_celsius"]
biomass = {t: run_benchmark(t) for t in temperatures}

fig, ax = plt.subplots(figsize=(0.70 * fs.WIDTH_FULL, 0.50 * fs.WIDTH_FULL))
colors = plt.cm.viridis(np.linspace(0, 0.9, len(temperatures)))
for t, color in zip(temperatures, colors, strict=True):
    series = biomass[t].isel(X=0, Y=0).squeeze().values
    days = np.arange(1, len(series) + 1)  # elapsed days (1-based for log scale)
    ax.plot(days, series, lw=1.6, color=color)
    ax.axhline(analytical_asymptote(t), ls="--", lw=1.4, color=color, alpha=0.85)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(1, DURATION_DAYS)   # 1 day to the last simulated day
ax.set_ylim(50, 10000)
plain = FuncFormatter(lambda value, _: f"{value:g}")  # 1, 10, 100, 1000 instead of 10^n
ax.xaxis.set_major_formatter(plain)
ax.yaxis.set_major_formatter(plain)
ax.set_xlabel("Elapsed time (days)")
ax.set_ylabel(r"Biomass (mg m$^{-2}$)")
ax.grid(True, which="both", alpha=0.3, linewidth=0.5)

# Legend: colour = temperature, dashed = analytical asymptote.
colour_handles = [Line2D([0], [0], color=c, lw=2, label=f"{t} °C") for t, c in zip(temperatures, colors, strict=True)]
asymptote_handle = Line2D([0], [0], color="black", lw=1.4, ls="--", label="Analytical asymptote")
ax.legend(handles=[*colour_handles, asymptote_handle], loc="lower right", ncol=2,
          frameon=True, framealpha=0.95, facecolor="white", edgecolor="0.7")

fig.tight_layout()
fs.save(fig, "Figure_3", subdir=None)
