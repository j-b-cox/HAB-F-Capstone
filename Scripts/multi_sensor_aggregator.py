#!/usr/bin/env python3
"""
multi_sensor_aggregator.py

Aggregates daily remote-sensing reflectance (Rrs) from multiple satellite sensors
over a specified date range and geographic bounding box. For each day, reflectances
from all available sensors are aligned to a common wavelength axis and averaged per
pixel. Missing bands in any sensor become NaNs during that day’s aggregation.

Exports:
    aggregate_sensors(start_date, end_date, bbox, sensors, resolution=0.01,
                      data_dir="../Data/", cache_dir="../Cache/")
"""

import os
import sys
import numpy as np
import xarray as xr
import earthaccess
from datetime import datetime, timedelta
import pickle
import time
import random
import logging
from tqdm import tqdm
from typing import List, Tuple, Dict

# ---------------------------
# Logging setup
# ---------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("multi_sensor_aggregator")


# ---------------------------
# Utility Functions
# ---------------------------

def parse_date_range(start_date: str, end_date: str) -> List[datetime]:
    """
    Parse YYYY-MM-DD strings into a list of datetime objects (inclusive).
    """
    sd = datetime.strptime(start_date, "%Y-%m-%d")
    ed = datetime.strptime(end_date, "%Y-%m-%d")
    num_days = (ed - sd).days + 1
    return [sd + timedelta(days=i) for i in range(num_days)]


def build_grid(
    bbox: Tuple[float, float, float, float],
    resolution: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build latitude/longitude bins and centers for a given bounding box and resolution.
    Returns (lat_bins, lon_bins, lat_centers, lon_centers).
    """
    lon_min, lat_min, lon_max, lat_max = bbox
    lat_bins = np.arange(lat_min, lat_max + resolution, resolution)
    lon_bins = np.arange(lon_min, lon_max + resolution, resolution)
    lat_centers = 0.5 * (lat_bins[:-1] + lat_bins[1:])
    lon_centers = 0.5 * (lon_bins[:-1] + lon_bins[1:])
    return lat_bins, lon_bins, lat_centers, lon_centers


def safe_search(
    short_name: str,
    temporal: Tuple[str, str],
    bounding_box: Tuple[float, float, float, float],
    max_retries: int = 500
) -> List[dict]:
    """
    Wraps earthaccess.search_data with retry logic.
    Returns list of granule metadata dicts or empty list on failure.
    """
    retries = 0
    while True:
        try:
            results = earthaccess.search_data(
                short_name=short_name,
                temporal=temporal,
                bounding_box=bounding_box
            )
            return results
        except Exception as e:
            retries += 1
            if retries >= max_retries:
                logger.error(f"Search for {short_name} {temporal} failed after {max_retries} retries: {e}")
                return []
            wait = 5 + random.uniform(0, 3)
            logger.warning(f"Search error ({short_name}, {temporal}): {e}. Retrying in {wait:.1f}s...")
            time.sleep(wait)


def safe_download(
    results: List[dict],
    directory: str = "../Data/",
    max_retries: int = 5
) -> List[str]:
    """
    Wraps earthaccess.download with retry logic.
    Returns list of local file paths downloaded, or empty list on failure.
    """
    retries = 0
    while True:
        try:
            paths = earthaccess.download(results, directory)
            return paths
        except Exception as e:
            retries += 1
            if retries >= max_retries:
                logger.error(f"Download failed after {max_retries} retries: {e}")
                return []
            wait = 5 + random.uniform(0, 3)
            logger.warning(f"Download error: {e}. Retrying in {wait:.1f}s...")
            time.sleep(wait)


# ---------------------------
# Reference Wavelength Retrieval
# ---------------------------

def get_reference_wavelengths(
    sensor: str,
    bbox: Tuple[float, float, float, float]
) -> np.ndarray:
    """
    Given a sensor short name, search for a nearby reference granule and extract its
    wavelength array. Supports:
      - “MODISA_L2_OC” and “MODIST_L2_OC”
      - “OLCIS3A_L2_EFR_OC” and “OLCIS3B_L2_EFR_OC”
      - “PACE_OCI_L2_AOP”
    Returns a 1D numpy array of wavelengths (nm).
    """
    if sensor in ["MODISA_L2_OC", "MODIST_L2_OC"]:
        candidates = ["MODISA_L2_OC", "MODIST_L2_OC"]
    elif sensor in ["OLCIS3A_L2_EFR_OC", "OLCIS3B_L2_EFR_OC"]:
        candidates = ["OLCIS3A_L2_EFR_OC", "OLCIS3B_L2_EFR_OC"]
    elif sensor == "PACE_OCI_L2_AOP":
        candidates = ["PACE_OCI_L2_AOP"]
    else:
        raise ValueError(f"Unsupported sensor: {sensor}")

    # Try up to three short windows to find at least one granule
    search_windows = [
        ("2024-06-01", "2024-06-05"),
        ("2024-05-01", "2024-06-01"),
        ("2024-04-01", "2024-05-01"),
    ]

    for sat in candidates:
        for temporal in search_windows:
            logger.info(f"Trying {sat} for wavelengths in window {temporal}…")
            ref_results = safe_search(sat, temporal, bbox)
            if not ref_results:
                continue

            # Download only the first match to inspect band parameters
            ref_file = safe_download(ref_results, "../Data/")[0]
            try:
                with xr.open_dataset(ref_file, group="sensor_band_parameters") as ds:
                    if "wavelength" in ds:
                        wave = ds["wavelength"].data
                    elif "wavelength_3d" in ds:
                        wave = ds["wavelength_3d"].data
                    else:
                        raise KeyError("No wavelength variable found in reference file.")
                logger.info(f"Retrieved {len(wave)} wavelengths from {ref_file}.")
                return wave
            except Exception as e:
                logger.warning(f"Failed to extract wavelengths from {ref_file}: {e}")

    raise RuntimeError(f"No reference granule found to retrieve wavelengths for {sensor}.")


# ---------------------------
# Granule Processing per Sensor
# ---------------------------

def process_granule_modis(
    path: str,
    wave_modis: np.ndarray,
    lat_bins: np.ndarray,
    lon_bins: np.ndarray,
    cache_dir: str,
    bbox: Tuple[float, float, float, float]
):
    """
    Extract MODIS L2 OC Rrs bands from a single .nc granule into a cache .npz file.
    The .npz contains arrays: lat_idx, lon_idx, ch_idx, val.
    """
    base = os.path.basename(path)
    cache_file = os.path.join(cache_dir, f"{base}.npz")
    if os.path.exists(cache_file):
        return

    logger.info(f"Processing MODIS granule: {base}")
    try:
        nav = xr.open_dataset(path, group="navigation_data")
        lat = nav["latitude"].values
        lon = nav["longitude"].values

        rrs_ds = xr.open_dataset(path, group="geophysical_data")
        lat_idx_list = []
        lon_idx_list = []
        ch_idx_list = []
        val_list = []

        for ch_idx, wl in enumerate(wave_modis):
            var_name = f"Rrs_{int(round(wl))}"
            if var_name not in rrs_ds:
                continue
            band = rrs_ds[var_name].values
            mask = (
                np.isfinite(band)
                & (lat >= bbox[1]) & (lat <= bbox[3])
                & (lon >= bbox[0]) & (lon <= bbox[2])
            )
            if not np.any(mask):
                continue

            lat_valid = lat[mask]
            lon_valid = lon[mask]
            val_valid = band[mask]

            lat_idx = np.searchsorted(lat_bins, lat_valid) - 1
            lon_idx = np.searchsorted(lon_bins, lon_valid) - 1

            lat_idx_list.extend(lat_idx.tolist())
            lon_idx_list.extend(lon_idx.tolist())
            ch_idx_list.extend([ch_idx] * len(val_valid))
            val_list.extend(val_valid.tolist())

        np.savez_compressed(
            cache_file,
            lat_idx=np.array(lat_idx_list, dtype=np.int16),
            lon_idx=np.array(lon_idx_list, dtype=np.int16),
            ch_idx=np.array(ch_idx_list, dtype=np.int16),
            val=np.array(val_list, dtype=np.float32),
        )
        logger.info(f"Saved MODIS cache: {cache_file}")

    except Exception as e:
        logger.error(f"Failed to process MODIS granule {base}: {e}")


def process_granule_s3(
    path: str,
    wave_s3: np.ndarray,
    lat_bins: np.ndarray,
    lon_bins: np.ndarray,
    cache_dir: str,
    bbox: Tuple[float, float, float, float]
):
    """
    Extract Sentinel-3 L2 EFR OC Rrs bands from a single .nc granule into a cache .npz file.
    """
    base = os.path.basename(path)
    cache_file = os.path.join(cache_dir, f"{base}.npz")
    if os.path.exists(cache_file):
        return

    logger.info(f"Processing Sentinel-3 granule: {base}")
    try:
        nav = xr.open_dataset(path, group="navigation_data")
        lat = nav["latitude"].values
        lon = nav["longitude"].values

        rrs_ds = xr.open_dataset(path, group="geophysical_data")
        lat_idx_list = []
        lon_idx_list = []
        ch_idx_list = []
        val_list = []

        for ch_idx, wl in enumerate(wave_s3):
            var_name = f"Rrs_{int(round(wl))}"
            if var_name not in rrs_ds:
                continue
            band = rrs_ds[var_name].values
            mask = (
                np.isfinite(band)
                & (lat >= bbox[1]) & (lat <= bbox[3])
                & (lon >= bbox[0]) & (lon <= bbox[2])
            )
            if not np.any(mask):
                continue

            lat_valid = lat[mask]
            lon_valid = lon[mask]
            val_valid = band[mask]

            lat_idx = np.searchsorted(lat_bins, lat_valid) - 1
            lon_idx = np.searchsorted(lon_bins, lon_valid) - 1

            lat_idx_list.extend(lat_idx.tolist())
            lon_idx_list.extend(lon_idx.tolist())
            ch_idx_list.extend([ch_idx] * len(val_valid))
            val_list.extend(val_valid.tolist())

        np.savez_compressed(
            cache_file,
            lat_idx=np.array(lat_idx_list, dtype=np.int16),
            lon_idx=np.array(lon_idx_list, dtype=np.int16),
            ch_idx=np.array(ch_idx_list, dtype=np.int16),
            val=np.array(val_list, dtype=np.float32),
        )
        logger.info(f"Saved Sentinel-3 cache: {cache_file}")

    except Exception as e:
        logger.error(f"Failed to process Sentinel-3 granule {base}: {e}")


def process_granule_pace(
    path: str,
    wave_pace: np.ndarray,
    lat_bins: np.ndarray,
    lon_bins: np.ndarray,
    cache_dir: str,
    bbox: Tuple[float, float, float, float]
):
    """
    Extract PACE L2 AOP Rrs bands from a single .nc granule into a cache .npz file.
    """
    base = os.path.basename(path)
    cache_file = os.path.join(cache_dir, f"{base}.npz")
    if os.path.exists(cache_file):
        return

    logger.info(f"Processing PACE granule: {base}")
    try:
        nav = xr.open_dataset(path, group="navigation_data")
        lat = nav["latitude"].values
        lon = nav["longitude"].values

        # Rrs is stored as a 3D variable with coordinate “wavelength_3d”
        rrs_ds = xr.open_dataset(path, group="geophysical_data")["Rrs"]
        rrs_ds = rrs_ds.assign_coords(wavelength_3d=wave_pace)

        lat_idx_list = []
        lon_idx_list = []
        ch_idx_list = []
        val_list = []

        for ch_idx, wl in enumerate(wave_pace):
            band = rrs_ds.sel(wavelength_3d=wl, method="nearest").values
            mask = (
                np.isfinite(band)
                & (lat >= bbox[1]) & (lat <= bbox[3])
                & (lon >= bbox[0]) & (lon <= bbox[2])
            )
            if not np.any(mask):
                continue

            lat_valid = lat[mask]
            lon_valid = lon[mask]
            val_valid = band[mask]

            lat_idx = np.searchsorted(lat_bins, lat_valid) - 1
            lon_idx = np.searchsorted(lon_bins, lon_valid) - 1

            lat_idx_list.extend(lat_idx.tolist())
            lon_idx_list.extend(lon_idx.tolist())
            ch_idx_list.extend([ch_idx] * len(val_valid))
            val_list.extend(val_valid.tolist())

        np.savez_compressed(
            cache_file,
            lat_idx=np.array(lat_idx_list, dtype=np.int16),
            lon_idx=np.array(lon_idx_list, dtype=np.int16),
            ch_idx=np.array(ch_idx_list, dtype=np.int16),
            val=np.array(val_list, dtype=np.float32),
        )
        logger.info(f"Saved PACE cache: {cache_file}")

    except Exception as e:
        logger.error(f"Failed to process PACE granule {base}: {e}")


# ---------------------------
# Load Daily Data per Sensor
# ---------------------------

def load_daily_sensor(
    date: datetime,
    sensor: str,
    bbox: Tuple[float, float, float, float],
    lat_bins: np.ndarray,
    lon_bins: np.ndarray,
    wave_dict: Dict[str, np.ndarray],
    cache_dir: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and average Rrs for a single sensor on a given date.
    Args:
        date: datetime for the day.
        sensor: sensor short name.
        bbox: (lon_min, lat_min, lon_max, lat_max).
        lat_bins, lon_bins: 1D arrays of bin edges.
        wave_dict: Dict mapping sensor to its wavelength array.
        cache_dir: Directory for caching .npz files.

    Returns:
        daily_avg: 3D array (nlat, nlon, n_bands_sensor) of daily-averaged reflectance.
        wavelengths: 1D array of that sensor’s wavelengths.
    """
    date_str = date.strftime("%Y-%m-%d")
    temporal = (date_str, date_str)
    all_granules: List[dict] = []

    # Determine which short_name(s) to search, which wavelength array to use, and which processor
    if sensor in ["MODISA_L2_OC", "MODIST_L2_OC"]:
        short_names = [sensor]
        wave_ref = wave_dict[sensor]
        process_fn = process_granule_modis

    elif sensor in ["OLCIS3A_L2_EFR_OC", "OLCIS3B_L2_EFR_OC"]:
        short_names = [sensor]
        wave_ref = wave_dict[sensor]
        process_fn = process_granule_s3

    elif sensor == "PACE_OCI_L2_AOP":
        short_names = ["PACE_OCI_L2_AOP"]
        wave_ref = wave_dict[sensor]
        process_fn = process_granule_pace

    else:
        raise ValueError(f"Unsupported sensor: {sensor}")

    # Search each relevant collection for that day
    for sn in short_names:
        results = safe_search(sn, temporal, bbox)
        all_granules.extend(results)

    nlat = len(lat_bins) - 1
    nlon = len(lon_bins) - 1
    n_bands = len(wave_ref)

    # If nothing found, return a full-NaN daily array
    if not all_granules:
        logger.warning(f"No granules for {sensor} on {date_str}. Returning all-NaNs.")
        return np.full((nlat, nlon, n_bands), np.nan, dtype=np.float32), wave_ref

    # Download any granules that haven’t already been cached
    to_download: List[dict] = []
    for granule in all_granules:
        granule_id = granule.get("granule_id") or \
            granule["umm"]["DataGranule"]["ArchiveAndDistributionInformation"][0]["Name"]
        base = os.path.basename(granule_id)
        cache_file = os.path.join(cache_dir, f"{base}.npz")
        if not os.path.exists(cache_file):
            to_download.append(granule)

    if to_download:
        logger.info(f"Downloading {len(to_download)} new granules for {sensor} on {date_str}…")
        paths = safe_download(to_download, "../Data/")
        if not paths:
            logger.error(f"Failed to download granules for {sensor} on {date_str}. Returning NaNs.")
            return np.full((nlat, nlon, n_bands), np.nan, dtype=np.float32), wave_ref
    else:
        logger.info(f"All granules for {sensor} on {date_str} already cached.")

    # Build a list of local .nc file paths
    granule_paths: List[str] = []
    for granule in all_granules:
        granule_id = granule.get("granule_id") or \
            granule["umm"]["DataGranule"]["ArchiveAndDistributionInformation"][0]["Name"]
        base = os.path.basename(granule_id)
        local_nc = os.path.join("../Data/", base)
        granule_paths.append(local_nc)

    # Accumulators: sum and count (per band, per pixel)
    sum_all = np.zeros((n_bands, nlat, nlon), dtype=np.float64)
    count_all = np.zeros((n_bands, nlat, nlon), dtype=np.int32)

    for path in granule_paths:
        base = os.path.basename(path)
        cache_file = os.path.join(cache_dir, f"{base}.npz")

        if not os.path.exists(cache_file):
            process_fn(path, wave_ref, lat_bins, lon_bins, cache_dir, bbox)

        if not os.path.exists(cache_file):
            logger.error(f"Cache missing for granule {base}. Skipping.")
            continue

        data = np.load(cache_file)
        lat_idx = data["lat_idx"]
        lon_idx = data["lon_idx"]
        ch_idx = data["ch_idx"]
        val = data["val"]

        # Accumulate
        for j in range(len(val)):
            li = lat_idx[j]
            lj = lon_idx[j]
            ci = ch_idx[j]
            if 0 <= li < nlat and 0 <= lj < nlon and 0 <= ci < n_bands:
                sum_all[ci, li, lj] += val[j]
                count_all[ci, li, lj] += 1

    # Compute daily average: sum / count, set zeros to NaN
    with np.errstate(invalid="ignore", divide="ignore"):
        daily_avg = sum_all / count_all
        daily_avg[count_all == 0] = np.nan

    # Transpose to (nlat, nlon, n_bands)
    daily_avg = np.transpose(daily_avg, (1, 2, 0)).astype(np.float32)
    return daily_avg, wave_ref


# ---------------------------
# Channel Alignment & Averaging
# ---------------------------

def unify_channels(
    source_arrays: Dict[str, np.ndarray],
    source_wavelengths: Dict[str, np.ndarray],
    combined_wavelengths: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Align each source’s 3D array (nlat × nlon × n_bands_src) to the combined wavelength dimension
    (nlat × nlon × n_bands_combined). Missing bands become NaNs.
    """
    aligned: Dict[str, np.ndarray] = {}
    # Grab nlat/nlon from any 3D array
    nlat, nlon, _ = next(iter(source_arrays.values())).shape
    n_combined = len(combined_wavelengths)

    for src, array3d in source_arrays.items():
        wave_src = source_wavelengths[src]
        # Build an index map: for each combined wavelength, find index in wave_src (or -1 if missing)
        map_indices = []
        for wl in combined_wavelengths:
            idx_matches = np.where(np.isclose(wave_src, wl, atol=1e-6))[0]
            map_indices.append(int(idx_matches[0]) if idx_matches.size > 0 else -1)

        aligned_array = np.full((nlat, nlon, n_combined), np.nan, dtype=np.float32)
        for i_comb, i_src in enumerate(map_indices):
            if i_src >= 0:
                aligned_array[:, :, i_comb] = array3d[:, :, i_src]
        aligned[src] = aligned_array

    return aligned


def average_across_sensors(arrays_list: List[np.ndarray]) -> np.ndarray:
    """
    Given a list of aligned 3D arrays (nlat, nlon, n_bands_combined),
    compute per-pixel/channel average ignoring NaNs.
    """
    stacked = np.stack(arrays_list, axis=0)  # (n_sensors, nlat, nlon, n_bands)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_array = np.nanmean(stacked, axis=0)
    return mean_array.astype(np.float32)


# ---------------------------
# Main Aggregation Function
# ---------------------------

def aggregate_sensors(
    start_date: str,
    end_date: str,
    bbox: Tuple[float, float, float, float],
    sensors: List[str],
    resolution: float = 0.01,
    data_dir: str = "../Data/",
    cache_dir: str = "../Cache/"
) -> Tuple[np.ndarray, Dict]:
    """
    Aggregate daily Rrs from multiple sensors over the given date range and bbox.
    Args:
        start_date: "YYYY-MM-DD"
        end_date:   "YYYY-MM-DD"
        bbox:       (lon_min, lat_min, lon_max, lat_max)
        sensors:    List of sensor short names (e.g., ["MODISA_L2_OC", "PACE_OCI_L2_AOP", ...])
        resolution: Grid resolution in degrees.
        data_dir:   Directory for raw downloads.
        cache_dir:  Directory for processed .npz cache files.

    Returns:
        ndarray_all: 4D numpy array of shape (n_days, nlat, nlon, n_bands_combined)
        metadata:    Dict with keys:
                      - "wavelengths": combined 1D wavelength array (nm),
                      - "lat": lat_centers,
                      - "lon": lon_centers,
                      - "date_list": list of date strings.
    """
    # Authenticate Earthaccess
    logger.info("Authenticating Earthdata…")
    _ = earthaccess.login(persist=True)

    # Ensure directories exist
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    # Build lat/lon grid
    lat_bins, lon_bins, lat_centers, lon_centers = build_grid(bbox, resolution)
    nlat, nlon = len(lat_centers), len(lon_centers)

    # Retrieve reference wavelengths for each sensor separately
    wave_dict: Dict[str, np.ndarray] = {}
    for sensor in sensors:
        wave = get_reference_wavelengths(sensor, bbox)
        # Exclude wavelengths > 2300 nm
        wave = wave[wave <= 2300]
        wave_dict[sensor] = wave
        logger.info(f"{sensor} wavelengths (<=2300 nm): {wave}")

    # Build combined wavelength list (union across all sensors)
    all_waves = [wave_dict[s] for s in sensors]
    combined_wavelengths = np.unique(np.concatenate(all_waves))
    logger.info(f"Combined wavelength count: {len(combined_wavelengths)}")

    # Prepare date list
    date_objs = parse_date_range(start_date, end_date)
    total_days = len(date_objs)
    date_list_str = [d.strftime("%Y-%m-%d") for d in date_objs]
    logger.info(f"Aggregating {total_days} days from {start_date} to {end_date}…")

    # Initialize final 4D array
    n_bands_combined = len(combined_wavelengths)
    ndarray_all = np.full(
        (total_days, nlat, nlon, n_bands_combined),
        np.nan,
        dtype=np.float32
    )

    # Loop over each date
    for day_idx, current_date in enumerate(tqdm(date_objs, desc="Processing dates")):
        date_str = current_date.strftime("%Y-%m-%d")
        logger.info(f"Processing {date_str} ({day_idx+1}/{total_days})")

        # 1) Load each sensor’s daily average into 3D array
        sensor_arrays: Dict[str, np.ndarray] = {}
        sensor_waves: Dict[str, np.ndarray] = {}
        for sensor in sensors:
            daily_arr, wave_ref = load_daily_sensor(
                date=current_date,
                sensor=sensor,
                bbox=bbox,
                lat_bins=lat_bins,
                lon_bins=lon_bins,
                wave_dict=wave_dict,
                cache_dir=cache_dir
            )
            sensor_arrays[sensor] = daily_arr  # (nlat, nlon, n_bands_sensor)
            sensor_waves[sensor] = wave_ref

        # 2) Align channels for each sensor to the combined wavelength axis
        aligned = unify_channels(sensor_arrays, sensor_waves, combined_wavelengths)

        # 3) Average across sensors (ignoring NaNs)
        daily_combined = average_across_sensors(list(aligned.values()))  # (nlat, nlon, n_bands_combined)

        # 4) Store into final array
        ndarray_all[day_idx, :, :, :] = daily_combined

    # Build metadata
    metadata = {
        "wavelengths": combined_wavelengths,
        "lat": lat_centers,
        "lon": lon_centers,
        "date_list": date_list_str
    }

    return ndarray_all, metadata

