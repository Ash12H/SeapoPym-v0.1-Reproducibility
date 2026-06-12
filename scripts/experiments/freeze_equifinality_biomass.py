"""Freeze the equifinality product — every restart's best-fit biomass at the CMA-ES parameters.

For each single station, runs SeapoPym at the best parameters of EVERY seeded restart and freezes,
over the display window, the biomass of each seed plus the twin target (pseudo-observation). Despite
the seed-to-seed spread in recovered parameters (recruitment equifinality at the warm stations), the
biomass curves all collapse onto the target -> many parameter sets, one biomass (RC-3 #22). MERGED is
not included.

Serial model runs (no Dask), one per (station, seed) -> CPU-heavy; do NOT run alongside a live
ensemble. Best params per seed are read from the FROZEN seed-ensemble product (the display contract).
Saved as a compact NetCDF array (zlib) -> small even for 20 members x 6 stations at daily resolution;
keeping every member (not just an envelope) leaves the door open to a range-area plot later.

Layout: dim `member` is sorted by cost, so member 0 is the best seed of each station.
Input  : frozen seed-ensemble product (paths.cmaes_product) + data/ forcing + pseudo-obs + coords
Output : products/equifinality_biomass{_metric}.nc
         vars: biomass(station, member, time), target(station, time), seed(station, member)
Run    : .venv/bin/python scripts/experiments/freeze_equifinality_biomass.py [--metric nrmse_mean] [--max-seeds N]
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import xarray as xr
from seapopym.configuration.no_transport import (
    FunctionalGroupParameter, FunctionalGroupUnit, FunctionalTypeParameter,
    KernelParameter, MigratoryTypeParameter, NoTransportConfiguration,
)
from seapopym.model.no_transport_model import NoTransportSpaceOptimizedLightModel

from seapopym_repro import experiment as exp, paths

ap = argparse.ArgumentParser()
ap.add_argument("--metric", default=paths.PRODUCTION_METRIC, help="cost metric whose frozen best params to run")
ap.add_argument("--max-seeds", type=int, default=None, help="cap seeds per station (default all; for quick previews)")
args = ap.parse_args()
METRIC = args.metric

ZOOM0, ZOOM1 = "2015-01-01", "2019-12-31"      # display window

d = pd.read_csv(paths.cmaes_product("seed_ensemble", METRIC))
coords = json.load(open(paths.DATA / "stations_coords.json"))
forcing = exp.build_forcing()                                  # same forcing for every run; build once
target = xr.open_zarr(paths.DATA / "pseudo_observations.zarr")["observed_biomass"]
SINGLES = [e for e in exp.EXPERIMENTS if e != "MERGED"]         # MERGED dropped from this figure


def run_model(p):
    """Run SeapoPym at parameter set p; return the windowed biomass grid (functional group 0)."""
    fg = FunctionalGroupParameter(functional_group=[FunctionalGroupUnit(
        name="zooplankton", energy_transfert=p["energy_transfert"],
        functional_type=FunctionalTypeParameter(
            lambda_temperature_0=p["lambda_temperature_0"], gamma_lambda_temperature=p["gamma_lambda_temperature"],
            tr_0=p["tr_0"], gamma_tr=p["gamma_tr"]),
        migratory_type=MigratoryTypeParameter(day_layer=0, night_layer=0))])
    config = NoTransportConfiguration(
        forcing=forcing, functional_group=fg, kernel=KernelParameter(biomass_solver="implicit"))
    with NoTransportSpaceOptimizedLightModel.from_configuration(configuration=config) as model:
        model.run()
        model.state.compute()
        return model.state.biomass.sel(T=slice(ZOOM0, ZOOM1), functional_group=0, drop=True).load()


def at(da, lat, time_index=None):
    """Biomass at a station's latitude as a Series; aligned onto `time_index` if given."""
    s = da.sel(Y=lat, X=0, method="nearest")
    ser = pd.Series(s.values, index=pd.to_datetime(s["T"].values))
    return ser.reindex(time_index, method="nearest") if time_index is not None else ser


time_index, biomass, targ, seeds = None, [], [], []
for sid in SINGLES:
    sub = d[d.experiment == sid].sort_values("best_nrmse")     # best (lowest cost) first -> member 0
    if args.max_seeds:
        sub = sub.head(args.max_seeds)
    lat = coords[sid]["lat"]
    tz = at(target.sel(T=slice(ZOOM0, ZOOM1)), lat)            # twin target sets the canonical time axis
    if time_index is None:
        time_index = tz.index
    targ.append(tz.reindex(time_index, method="nearest").to_numpy())
    print(f"  {sid}: {len(sub)} seeds (best=seed {int(sub.iloc[0].seed)}) ...", flush=True)
    members = [at(run_model({k: float(r[k]) for k in exp.PARAM_KEYS}), lat, time_index).to_numpy()
               for _, r in sub.iterrows()]
    biomass.append(np.array(members))
    seeds.append(sub.seed.astype(int).to_numpy())

ds = xr.Dataset(
    {"biomass": (("station", "member", "time"), np.array(biomass)),
     "target": (("station", "time"), np.array(targ)),
     "seed": (("station", "member"), np.array(seeds))},
    coords={"station": SINGLES, "member": np.arange(np.array(biomass).shape[1]), "time": time_index},
    attrs={"metric": METRIC, "note": "member 0 = best seed (lowest cost); biomass in g C m-2"})
out = paths.PRODUCTS / f"equifinality_biomass{paths.metric_tag(METRIC)}.nc"
ds.to_netcdf(out, encoding={"biomass": {"zlib": True, "complevel": 4},
                            "target": {"zlib": True, "complevel": 4}})
print(f"froze {out.name} ({ds.sizes['station']} stations x {ds.sizes['member']} members x {ds.sizes['time']} days)", flush=True)
