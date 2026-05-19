"""download_cmems_global.py — Self-contained CMEMS-LMTL global download.

Downloads the 5 variables used by this paper from the Copernicus Marine LMTL
product, regrids from native 1/12° to 1° (bin-averaging), and writes the
canonical Zarr `data/forcings_global.zarr` directly.

This is the heavy companion of `scripts/download_cmems_stations.py`. The two
scripts share the same regridding algorithm; running this one on the full
globe gives results that are bit-for-bit identical to the stations script at
the six station coordinates (verified on January 2010).

Source DOI (CMEMS LMTL product): 10.48670/moi-00020

Variables and provenance:
    - temperature: sub-dataset `cmems_mod_glo_bgc_my_0.083deg-lmtl-Fphy_P1D-i`,
      variable `T`, depth index 1 (epipelagic). Renamed `temperature`.
    - U, V: same sub-dataset, depth index 1 (epipelagic, for current-norm map).
    - npp, zooc: sub-dataset `cmems_mod_glo_bgc_my_0.083deg-lmtl_P1D-i`.

Resumable: the script appends to `data/forcings_global.zarr` month by month
and skips months that are already present.

Authentication
--------------
Requires Copernicus Marine credentials (see README.md, section Quick start).

Output
------
    data/forcings_global.zarr — Dataset with dims (T=8035, Y=170, X=360),
    variables temperature, U, V, npp, zooc. Approximate size on disk: ~6 GB.

Runtime
-------
A few hours on a residential connection; bandwidth-bound rather than
CPU-bound. Allow a few minutes per month with a fast link.

Usage
-----
    python scripts/download_cmems_global.py
    python scripts/download_cmems_global.py --start-date 2010-01-01 --end-date 2010-01-31
"""

from __future__ import annotations

import argparse
import calendar
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
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
DEFAULT_OUTPUT = ROOT_DIR / "data" / "forcings_global.zarr"

# CMEMS dataset IDs (LMTL product, DOI 10.48670/moi-00020)
FPHY_DATASET = "cmems_mod_glo_bgc_my_0.083deg-lmtl-Fphy_P1D-i"
BIO_DATASET = "cmems_mod_glo_bgc_my_0.083deg-lmtl_P1D-i"

# Epipelagic depth value in the source product (the LMTL product uses 1, 2, 3
# for epi, upper-meso, lower-meso respectively).
EPI_DEPTH = 1

# Default time range matches Section 2.3 of the manuscript.
DEFAULT_START_DATE = "1998-01-01"
DEFAULT_END_DATE = "2019-12-31"

# Target 1° grid edges (identical to the global pipeline, Section 2.3).
TARGET_LAT_EDGES = np.arange(-85, 86, 1.0)
TARGET_LON_EDGES = np.arange(-180, 181, 1.0)
TARGET_LAT = (0.5 * (TARGET_LAT_EDGES[:-1] + TARGET_LAT_EDGES[1:])).astype(np.float32)
TARGET_LON = (0.5 * (TARGET_LON_EDGES[:-1] + TARGET_LON_EDGES[1:])).astype(np.float32)

# CMEMS Fphy product bounding box (extends slightly past +/-180, +/-90).
DOMAIN_BBOX = dict(
    minimum_longitude=-180.0,
    maximum_longitude=179.917,
    minimum_latitude=-80.0,
    maximum_latitude=89.917,
)

# Variable attributes written to the output Zarr.
VARIABLE_ATTRS = {
    "temperature": {"units": "degC",       "long_name": "Mean temperature in the epipelagic layer"},
    "U":           {"units": "m/s",        "long_name": "Zonal current velocity (epipelagic layer)"},
    "V":           {"units": "m/s",        "long_name": "Meridional current velocity (epipelagic layer)"},
    "npp":         {"units": "mg/m^2/day", "long_name": "Vertically integrated net primary production (VGPM)"},
    "zooc":        {"units": "g/m^2",      "long_name": "Zooplankton biomass (SEAPODYM-LMTL reference)"},
}
COORD_ATTRS = {
    "T": {"axis": "T"},
    "Y": {"axis": "Y", "units": "degrees_north", "standard_name": "latitude"},
    "X": {"axis": "X", "units": "degrees_east",  "standard_name": "longitude"},
}
DATASET_ATTRS = {
    "title": "SEAPODYM-LMTL global forcings and reference biomass (1 deg grid, 1998-2019)",
    "source": (
        "CMEMS product GLOBAL_MULTIYEAR_BGC_001_033 (DOI: 10.48670/moi-00020), "
        "downloaded at native 1/12 deg resolution and bin-averaged to 1 deg, "
        "following the procedure described in Section 2.3 of Lehodey et al. (2026, GMD)."
    ),
}


# =============================================================================
# Bin-averaging from native 1/12 deg to 1 deg
# =============================================================================


def regrid_to_1deg(da: xr.DataArray) -> np.ndarray:
    """Bin-average a (time, latitude, longitude) DataArray onto the 1° target grid."""
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

    data = da.values  # (time, latitude, longitude)
    n_t = data.shape[0]
    out = np.full((n_t, len(TARGET_LAT), len(TARGET_LON)), np.nan, dtype=np.float32)

    for ty, src_y_idx in lat_groups.items():
        for tx, src_x_idx in lon_groups.items():
            block = data[:, src_y_idx, :][:, :, src_x_idx]
            with np.errstate(all="ignore"):
                out[:, ty, tx] = np.nanmean(block.reshape(n_t, -1), axis=1)
    return out


# =============================================================================
# Per-variable, per-month download helpers
# =============================================================================


def _month_range(year: int, month: int) -> tuple[str, str]:
    last_day = calendar.monthrange(year, month)[1]
    return f"{year}-{month:02d}-01", f"{year}-{month:02d}-{last_day:02d}"


def _download_fphy(var: str, start: str, end: str) -> np.ndarray:
    """Download one Fphy variable (epipelagic only) for a time range and regrid."""
    log.info("    Fphy %s ...", var)
    ds = copernicusmarine.open_dataset(
        dataset_id=FPHY_DATASET,
        variables=[var],
        minimum_depth=EPI_DEPTH,
        maximum_depth=EPI_DEPTH,
        **DOMAIN_BBOX,
        start_datetime=start,
        end_datetime=end,
    ).load()
    arr = ds[var].squeeze("depth", drop=True) if "depth" in ds[var].dims else ds[var]
    regridded = regrid_to_1deg(arr)
    ds.close()
    return regridded


def _download_bio(var: str, start: str, end: str) -> np.ndarray:
    """Download one Bio variable (no depth dim) for a time range and regrid."""
    log.info("    Bio %s ...", var)
    ds = copernicusmarine.open_dataset(
        dataset_id=BIO_DATASET,
        variables=[var],
        **DOMAIN_BBOX,
        start_datetime=start,
        end_datetime=end,
    ).load()
    regridded = regrid_to_1deg(ds[var])
    ds.close()
    return regridded


def _download_one_month(year: int, month: int) -> tuple[xr.Dataset, int]:
    start, end = _month_range(year, month)
    n_days = calendar.monthrange(year, month)[1]
    log.info("[%04d-%02d] downloading (parallel)...", year, month)

    futures: dict = {}
    with ThreadPoolExecutor(max_workers=5) as pool:
        for var in ("T", "U", "V"):
            futures[pool.submit(_download_fphy, var, start, end)] = ("fphy", var)
        for var in ("npp", "zooc"):
            futures[pool.submit(_download_bio, var, start, end)] = ("bio", var)
        results: dict[str, np.ndarray] = {}
        for fut in as_completed(futures):
            kind, var = futures[fut]
            results[var] = fut.result()

    dates = pd.date_range(start, periods=n_days, freq="D")
    return xr.Dataset(
        {
            "temperature": (("T", "Y", "X"), results["T"][:n_days]),
            "U":           (("T", "Y", "X"), results["U"][:n_days]),
            "V":           (("T", "Y", "X"), results["V"][:n_days]),
            "npp":         (("T", "Y", "X"), results["npp"][:n_days]),
            "zooc":        (("T", "Y", "X"), results["zooc"][:n_days]),
        },
        coords={"T": dates, "Y": TARGET_LAT, "X": TARGET_LON},
    ), n_days


# =============================================================================
# Zarr I/O — resumable append
# =============================================================================


def _last_date(output: Path) -> pd.Timestamp | None:
    if not output.exists():
        return None
    try:
        with xr.open_zarr(output) as ds:
            return pd.Timestamp(ds["T"].values[-1])
    except Exception:
        return None


def _write(ds: xr.Dataset, output: Path) -> None:
    if not output.exists():
        for var, attrs in VARIABLE_ATTRS.items():
            if var in ds.data_vars:
                ds[var].attrs.update(attrs)
        for coord, attrs in COORD_ATTRS.items():
            if coord in ds.coords:
                ds[coord].attrs.update(attrs)
        ds.attrs.update(DATASET_ATTRS)
        encoding = {v: {"chunks": (31, len(TARGET_LAT), len(TARGET_LON))} for v in ds.data_vars}
        ds.to_zarr(output, mode="w", encoding=encoding, zarr_format=2)
    else:
        ds.to_zarr(output, append_dim="T")


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    parser.add_argument("--start-date", default=DEFAULT_START_DATE, help="Inclusive start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", default=DEFAULT_END_DATE, help="Inclusive end date (YYYY-MM-DD)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help=f"Output Zarr path (default: {DEFAULT_OUTPUT})")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    log.info("Output: %s", args.output)

    last = _last_date(args.output)
    if last is not None:
        log.info("Resuming — last date in zarr: %s", last.strftime("%Y-%m-%d"))

    start_ts = pd.Timestamp(args.start_date)
    end_ts = pd.Timestamp(args.end_date)
    months = pd.date_range(start_ts.replace(day=1), end_ts, freq="MS")
    total = len(months)

    for idx, m in enumerate(months, start=1):
        month_end = pd.Timestamp(m.year, m.month, calendar.monthrange(m.year, m.month)[1])
        if last is not None and month_end <= last:
            log.info("[%d/%d] %s — skipped (already in zarr)", idx, total, m.strftime("%Y-%m"))
            continue
        ds, n_days = _download_one_month(m.year, m.month)
        _write(ds, args.output)
        log.info("[%d/%d] %s — wrote %d days", idx, total, m.strftime("%Y-%m"), n_days)

    with xr.open_zarr(args.output) as ds:
        size_gb = sum(f.stat().st_size for f in args.output.rglob("*") if f.is_file()) / 1e9
        log.info("=== Done ===")
        log.info("Time: %s -> %s (%d days)", str(ds["T"].values[0])[:10], str(ds["T"].values[-1])[:10], ds.sizes["T"])
        log.info("Variables: %s", list(ds.data_vars))
        log.info("Size on disk: %.2f GB", size_gb)


if __name__ == "__main__":
    main()
