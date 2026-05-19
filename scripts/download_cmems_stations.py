"""download_cmems_stations.py — Download CMEMS forcings for the 6 oceanographic stations.

Faithful to the procedure described in Section 2.3 of Lehodey et al. (2026, GMD):

    1. Download CMEMS LMTL product (GLOBAL_MULTIYEAR_BGC_001_033) at native 1/12° resolution
    2. Regrid to 1° using bin-averaging (same algorithm as the global pipeline)
    3. Extract the 1° cell containing each station coordinate

Variables:
    - temperature: epipelagic layer (depth index 1), from Fphy product
    - npp:         vertically integrated primary production, from Bio product
    - zooc:        zooplankton biomass (LMTL reference), from Bio product

Stations (manuscript Table 2):
    BARENTS  75.0°N   40.0°E   — Barents Sea
    PAPA     50.0°N  132.0°W   — Station Papa
    BISCAY   45.5°N    4.0°W   — Bay of Biscay
    BATS     32.0°N   64.0°W   — Bermuda Atlantic Time-series Study
    CANARY   30.0°N   13.0°W   — Canary Islands
    HOT      23.0°N  158.0°W   — Hawaii Ocean Time-series

Source DOI: 10.48670/moi-00020

Authentication
--------------
Requires Copernicus Marine credentials. Either:
    - Run `copernicusmarine login` once interactively
    - Or set COPERNICUSMARINE_SERVICE_USERNAME / COPERNICUSMARINE_SERVICE_PASSWORD env vars

Output
------
    data/stations.zarr — Dataset with dims (time, station) and variables
    (temperature, npp, zooc). Approximate size: 5-15 MB.

Usage
-----
    python scripts/download_cmems_stations.py

This is a lightweight equivalent of the full global download procedure used in the
manuscript. To reproduce the global figures (Fig. 1, Fig. 4), extend the bounding box
to the full domain as described in Section 2.3 (see the companion full-global script in
the SeapoPym Article 2 codebase, which uses the same regridding algorithm).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import copernicusmarine
import numpy as np
import pandas as pd
import xarray as xr

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
STATIONS_JSON = DATA_DIR / "stations_coords.json"
OUTPUT_ZARR = DATA_DIR / "stations.zarr"

# CMEMS dataset IDs (LMTL product, DOI 10.48670/moi-00020)
FPHY_DATASET = "cmems_mod_glo_bgc_my_0.083deg-lmtl-Fphy_P1D-i"
BIO_DATASET = "cmems_mod_glo_bgc_my_0.083deg-lmtl_P1D-i"

# Epipelagic layer (depth index 1 in the LMTL product convention)
EPI_DEPTH = 1

# Default time range (Section 2.3 of the manuscript); can be overridden via CLI
DEFAULT_START_DATE = "1998-01-01"
DEFAULT_END_DATE = "2019-12-31"

# Half-width of the bbox around each station (degrees). Must cover the full 1° cell
# containing the station coordinate. 1° is a safe margin for any rounding edge case.
BBOX_HALF = 1.0

# Target 1° grid edges — identical to the global pipeline (Section 2.3 of the manuscript)
TARGET_LAT_EDGES = np.arange(-85, 86, 1.0)
TARGET_LON_EDGES = np.arange(-180, 181, 1.0)
TARGET_LAT = 0.5 * (TARGET_LAT_EDGES[:-1] + TARGET_LAT_EDGES[1:])
TARGET_LON = 0.5 * (TARGET_LON_EDGES[:-1] + TARGET_LON_EDGES[1:])


# =============================================================================
# Regridding (bin-averaging to 1°, matches the global pipeline)
# =============================================================================


def regrid_bbox_to_1deg(da: xr.DataArray) -> xr.DataArray:
    """Bin-average a source DataArray onto the global 1° target grid.

    Only target cells that contain at least one source pixel are returned.
    Expects dims (time, latitude, longitude).
    """
    src_lat = da["latitude"].values
    src_lon = da["longitude"].values

    iy_bin = np.clip(np.digitize(src_lat, TARGET_LAT_EDGES) - 1, 0, len(TARGET_LAT) - 1)
    ix_bin = np.clip(np.digitize(src_lon, TARGET_LON_EDGES) - 1, 0, len(TARGET_LON) - 1)

    lat_groups: dict[int, list[int]] = {}
    for j, b in enumerate(iy_bin):
        lat_groups.setdefault(int(b), []).append(j)
    lon_groups: dict[int, list[int]] = {}
    for j, b in enumerate(ix_bin):
        lon_groups.setdefault(int(b), []).append(j)

    target_lats_idx = sorted(lat_groups.keys())
    target_lons_idx = sorted(lon_groups.keys())

    data = da.values  # (T, Y, X)
    n_t = data.shape[0]
    out = np.full((n_t, len(target_lats_idx), len(target_lons_idx)), np.nan, dtype=np.float32)

    for ty, lat_idx in enumerate(target_lats_idx):
        src_y = lat_groups[lat_idx]
        for tx, lon_idx in enumerate(target_lons_idx):
            src_x = lon_groups[lon_idx]
            block = data[:, src_y, :][:, :, src_x]
            with np.errstate(all="ignore"):
                out[:, ty, tx] = np.nanmean(block.reshape(n_t, -1), axis=1)

    return xr.DataArray(
        out,
        dims=("time", "lat", "lon"),
        coords={
            "time": da["time"],
            "lat": ("lat", np.array([TARGET_LAT[i] for i in target_lats_idx], dtype=np.float32)),
            "lon": ("lon", np.array([TARGET_LON[i] for i in target_lons_idx], dtype=np.float32)),
        },
    )


def extract_station_cell(da: xr.DataArray, lat: float, lon: float) -> xr.DataArray:
    """Extract the 1° cell whose centre is nearest to (lat, lon)."""
    return da.sel(lat=lat, lon=lon, method="nearest", drop=True)


# =============================================================================
# Per-station download
# =============================================================================


def download_station(name: str, lat: float, lon: float, start: str, end: str) -> xr.Dataset:
    """Download T, npp, zooc for one station and regrid to its 1° cell."""
    log.info(
        "[%s] bbox lat=[%.1f, %.1f], lon=[%.1f, %.1f] | %s -> %s",
        name, lat - BBOX_HALF, lat + BBOX_HALF, lon - BBOX_HALF, lon + BBOX_HALF, start, end,
    )

    bbox = dict(
        minimum_longitude=lon - BBOX_HALF,
        maximum_longitude=lon + BBOX_HALF,
        minimum_latitude=lat - BBOX_HALF,
        maximum_latitude=lat + BBOX_HALF,
        start_datetime=start,
        end_datetime=end,
    )

    log.info("[%s]   Fetching temperature (Fphy, depth=%d)...", name, EPI_DEPTH)
    ds_t = copernicusmarine.open_dataset(
        dataset_id=FPHY_DATASET,
        variables=["T"],
        minimum_depth=EPI_DEPTH,
        maximum_depth=EPI_DEPTH,
        **bbox,
    ).load()

    log.info("[%s]   Fetching npp and zooc (Bio)...", name)
    ds_bio = copernicusmarine.open_dataset(
        dataset_id=BIO_DATASET,
        variables=["npp", "zooc"],
        **bbox,
    ).load()

    log.info("[%s]   Regridding to 1° and extracting station cell...", name)
    t_da = ds_t["T"].squeeze("depth", drop=True)
    t_cell = extract_station_cell(regrid_bbox_to_1deg(t_da), lat, lon)
    npp_cell = extract_station_cell(regrid_bbox_to_1deg(ds_bio["npp"]), lat, lon)
    zooc_cell = extract_station_cell(regrid_bbox_to_1deg(ds_bio["zooc"]), lat, lon)

    ds_t.close()
    ds_bio.close()

    return xr.Dataset(
        {
            "temperature": t_cell.astype(np.float32),
            "npp": npp_cell.astype(np.float32),
            "zooc": zooc_cell.astype(np.float32),
        }
    )


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    parser.add_argument("--start-date", default=DEFAULT_START_DATE, help="Inclusive start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", default=DEFAULT_END_DATE, help="Inclusive end date (YYYY-MM-DD)")
    parser.add_argument("--output", type=Path, default=OUTPUT_ZARR, help="Output Zarr path")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    with open(STATIONS_JSON) as f:
        stations = json.load(f)
    log.info("Loaded %d stations from %s", len(stations), STATIONS_JSON.name)

    per_station = {}
    for name, info in stations.items():
        ds = download_station(name, info["lat"], info["lon"], args.start_date, args.end_date)
        per_station[name] = ds

    log.info("Combining stations into one Dataset...")
    combined = xr.concat(
        [
            per_station[name]
            .expand_dims(station=[name])
            .assign_coords(
                station_lat=("station", [stations[name]["lat"]]),
                station_lon=("station", [stations[name]["lon"]]),
            )
            for name in stations
        ],
        dim="station",
    )

    if args.output.exists():
        log.warning("Output Zarr already exists; overwriting %s", args.output)
    combined.to_zarr(args.output, mode="w", zarr_format=2)

    size_mb = sum(f.stat().st_size for f in args.output.rglob("*") if f.is_file()) / 1e6
    log.info("=== Done ===")
    log.info("Output: %s (%.1f MB)", args.output, size_mb)
    log.info("Stations: %s", list(combined.station.values))
    log.info("Variables: %s", list(combined.data_vars))
    log.info(
        "Time: %s → %s (%d days)",
        pd.Timestamp(combined.time.values[0]).date(),
        pd.Timestamp(combined.time.values[-1]).date(),
        combined.sizes["time"],
    )


if __name__ == "__main__":
    main()
