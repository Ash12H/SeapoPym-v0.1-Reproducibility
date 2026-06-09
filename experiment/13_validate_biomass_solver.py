"""Validate the configurable biomass solver (explicit vs implicit IMEX).

Two checks, both under constant forcing so the analytical steady state B_eq = E*NPP/lambda(T)
is known (lambda uses the model's internal Gillooly temperature transform, as in fig03):

  1. STABLE regime — reference params, T in {0,10,20,30} C:
     both solvers must converge to B_eq AND agree with each other (difference O(dt)).
     -> proves the implicit scheme is correct (not a "fake" implicit) and matches the
        published explicit results where the explicit scheme is valid.

  2. STIFF regime — warm T with the GA-bound mortality corner (lambda0=0.1, gamma_lambda=0.25):
     lambda*dt >> 2, so the explicit Euler scheme is CFL-unstable. The explicit run is
     expected to blow up (non-finite) or oscillate negative; the implicit run must stay
     finite, positive, and converge to B_eq.
     -> proves the implicit scheme makes the full GA mortality bounds safe at warm stations,
        which is the whole point of re-running Sobol on the unified GA bounds.

Run: .venv/bin/python experiment/13_validate_biomass_solver.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml
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

ROOT = Path(__file__).resolve().parents[1]
PARAMS = yaml.safe_load(open(ROOT / "parameters.yaml"))
REF = PARAMS["model_parameters"]["reference"]
NPP = PARAMS["theoretical_benchmark"]["npp_constant"]


def transform_temperature(t: float) -> float:
    """Model's internal physiological temperature transform (Gillooly kernel)."""
    return t / (1 + t / 273)


def asymptote(t_celsius: float, lambda0: float, gamma_lambda: float) -> float:
    lam = lambda0 * np.exp(gamma_lambda * transform_temperature(t_celsius))
    return REF["energy_transfert"] * NPP / lam


def build_forcing(t_celsius: float, duration_days: int):
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
        np.full((duration_days, 1, 1), NPP, dtype=np.float32),
        dims=("time", "latitude", "longitude"),
        coords={"time": time, "latitude": [0.5], "longitude": [0.5]},
        attrs={"units": "mg/m2/day"},
    )
    for da in (temperature, npp):
        for name, attrs in coord_attrs.items():
            if name in da.coords:
                da[name].attrs.update(attrs)
    return temperature, npp


def run(t_celsius: float, solver: str, lambda0: float, gamma_lambda: float, duration_days: int) -> np.ndarray:
    temperature, npp = build_forcing(t_celsius, duration_days)
    fg = FunctionalGroupParameter(functional_group=[FunctionalGroupUnit(
        name="test",
        energy_transfert=REF["energy_transfert"],
        functional_type=FunctionalTypeParameter(
            lambda_temperature_0=lambda0, gamma_lambda_temperature=gamma_lambda,
            tr_0=REF["tr_0"], gamma_tr=REF["gamma_tr"],
        ),
        migratory_type=MigratoryTypeParameter(day_layer=1, night_layer=1),
    )])
    config = NoTransportConfiguration(
        forcing=ForcingParameter(
            temperature=ForcingUnit(forcing=temperature),
            primary_production=ForcingUnit(forcing=npp),
        ),
        functional_group=fg,
        kernel=KernelParameter(biomass_solver=solver),
    )
    with NoTransportModel.from_configuration(configuration=config) as model:
        model.run()
        b = model.state.biomass.copy().pint.quantify().pint.to("mg/m2").pint.dequantify()
    return b.isel(X=0, Y=0).squeeze().values


# ---------------------------------------------------------------- check 1: stable regime
print("=" * 78)
print("CHECK 1 — stable regime (reference params): convergence + solver agreement")
print("=" * 78)
l0, gl = REF["lambda_temperature_0"], REF["gamma_lambda_temperature"]
DUR = 1500  # long enough that even T=0 (1/lambda=150 d) reaches steady state
TAIL = 60   # the two schemes must agree on the steady tail (not during the transient)
print(f"{'T(C)':>5}{'B_eq':>12}{'expl err%':>11}{'impl err%':>11}"
      f"{'tail|e-i|%':>12}{'peak|e-i|':>11}{'@day':>7}")
ok1 = True
for t in [0, 10, 20, 30]:
    beq = asymptote(t, l0, gl)
    be = run(t, "explicit", l0, gl, DUR)
    bi = run(t, "implicit", l0, gl, DUR)
    ee = abs(be[-1] - beq) / beq * 100
    ei = abs(bi[-1] - beq) / beq * 100
    n = min(len(be), len(bi))
    d = np.abs(be[:n] - bi[:n])
    tail = np.nanmax(d[-TAIL:]) / beq * 100         # steady-tail relative disagreement
    peak, peak_day = float(np.nanmax(d)), int(np.nanargmax(d) + 1)  # transient peak (diagnostic)
    print(f"{t:>5}{beq:>12.2f}{ee:>11.3f}{ei:>11.3f}{tail:>12.4f}{peak:>11.2f}{peak_day:>7}")
    ok1 &= (ee < 0.1) and (ei < 0.1) and (tail < 0.01) and np.isfinite(be).all() and np.isfinite(bi).all()
print(f"  -> both converge to B_eq (<0.1%) and agree on steady tail (<0.01%): {'PASS' if ok1 else 'FAIL'}")

# ---------------------------------------------------------------- check 2: stiff regime
print("\n" + "=" * 78)
print("CHECK 2 — stiff regime: GA-bound mortality corner at a warm station")
print("=" * 78)
l0_stiff = PARAMS["model_parameters"]["bounds"]["lambda_temperature_0"][1]      # 0.1
gl_stiff = PARAMS["model_parameters"]["bounds"]["gamma_lambda_temperature"][1]  # 0.25
t_warm = 25.0
lam_eff = l0_stiff * np.exp(gl_stiff * transform_temperature(t_warm))
print(f"T={t_warm} C, lambda0={l0_stiff}, gamma_lambda={gl_stiff}  "
      f"->  lambda_eff={lam_eff:.2f}/day, lambda*dt={lam_eff:.2f} (stable iff <2)")
beq = asymptote(t_warm, l0_stiff, gl_stiff)
be = run(t_warm, "explicit", l0_stiff, gl_stiff, 365)
bi = run(t_warm, "implicit", l0_stiff, gl_stiff, 365)
print(f"  B_eq (analytical)      : {beq:.4f} mg/m2")
print(f"  explicit: finite={np.isfinite(be).all()}  min={np.nanmin(be):.3e}  "
      f"max={np.nanmax(be):.3e}  final={be[-1]:.3e}")
print(f"  implicit: finite={np.isfinite(bi).all()}  min={np.nanmin(bi):.3e}  "
      f"max={np.nanmax(bi):.3e}  final={bi[-1]:.4f}")
expl_broken = (not np.isfinite(be).all()) or (np.nanmin(be) < 0) or (np.nanmax(be) > 100 * beq)
impl_ok = np.isfinite(bi).all() and (np.nanmin(bi) >= -1e-9) and (abs(bi[-1] - beq) / beq < 0.05)
print(f"  -> explicit unstable as expected: {'YES' if expl_broken else 'NO'}")
print(f"  -> implicit stable & converges to B_eq: {'PASS' if impl_ok else 'FAIL'}")

print("\n" + "=" * 78)
print(f"OVERALL: {'PASS — implicit solver validated' if (ok1 and impl_ok and expl_broken) else 'REVIEW NEEDED'}")
print("=" * 78)
