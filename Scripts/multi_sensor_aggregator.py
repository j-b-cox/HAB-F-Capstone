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
import time
import pickle
import random
import logging
import earthaccess

import numpy  as np
import pandas as pd
import xarray as xr

from tqdm     import tqdm
from netCDF4  import Dataset
from datetime import datetime, timedelta
from typing   import List, Tuple, Dict, Union
from scipy.interpolate import interp1d

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
    sd = datetime.strptime(start_date, "%Y-%m-%d")
    ed = datetime.strptime(end_date, "%Y-%m-%d")
    num_days = (ed - sd).days + 1
    return [sd + timedelta(days=i) for i in range(num_days)]


def build_grid(
    bbox: Tuple[float, float, float, float],
    resolution: Union[float, Tuple[float, float]]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build grid bin edges and centers for given bbox and resolution.
    resolution: either a single float (deg for both lat and lon) or a tuple (res_lat_deg, res_lon_deg).
    bbox: (lon_min, lat_min, lon_max, lat_max).
    Returns: (lat_bins, lon_bins, lat_centers, lon_centers).
    """
    lon_min, lat_min, lon_max, lat_max = bbox
    if isinstance(resolution, tuple) or isinstance(resolution, list):
        res_lat, res_lon = resolution
    else:
        res_lat = res_lon = resolution

    # Build edges so that centers are spaced by res_lat / res_lon
    lat_bins = np.arange(lat_min, lat_max + res_lat*0.5, res_lat)
    lon_bins = np.arange(lon_min, lon_max + res_lon*0.5, res_lon)
    # centers between bins
    lat_centers = 0.5 * (lat_bins[:-1] + lat_bins[1:])
    lon_centers = 0.5 * (lon_bins[:-1] + lon_bins[1:])
    return lat_bins, lon_bins, lat_centers, lon_centers


def bbox_to_str(bbox: Tuple[float, float, float, float]) -> str:
    """Convert bounding box to standardized string with 5 decimal places."""
    return "_".join(f"{coord:.5f}" for coord in bbox)


def safe_search(
    short_name: str,
    temporal: Tuple[str, str],
    bounding_box: Tuple[float, float, float, float],
    max_retries: int = 500
) -> List[dict]:
    retries = 0
    while True:
        try:
            return earthaccess.search_data(
                short_name=short_name,
                temporal=temporal,
                bounding_box=bounding_box
            )
        except Exception as e:
            retries += 1
            if retries >= max_retries:
                logger.error(f"Search for {short_name} {temporal} failed: {e}")
                return []
            time.sleep(5 + random.uniform(0, 3))


def safe_download(
    results: List[dict],
    directory: str = "../Data/",
    max_retries: int = 5
) -> List[str]:
    retries = 0
    while True:
        try:
            return earthaccess.download(results, directory)
        except Exception as e:
            retries += 1
            if retries >= max_retries:
                logger.error(f"Download failed: {e}")
                return []
            time.sleep(5 + random.uniform(0, 3))


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

def process_granule_modis(path: str, bbox: tuple) -> (np.ndarray, np.ndarray):
    """
    Process a MODIS granule: mask fill, apply scale/offset, average over bbox.
    Returns:
      native_wls: 1D array of band wavelengths (floats)
      native_means: 1D array of mean reflectance at those wavelengths
    """
    try:
        ds = Dataset(path, "r")
        lat = ds.groups['navigation_data'].variables['latitude'][:]
        lon = ds.groups['navigation_data'].variables['longitude'][:]

        lon_min, lat_min, lon_max, lat_max = bbox
        region_mask = (
            (lat >= lat_min) & (lat <= lat_max) &
            (lon >= lon_min) & (lon <= lon_max)
        )
        if not np.any(region_mask):
            ds.close()
            return np.array([]), np.array([])

        group = ds.groups['geophysical_data']
        # Gather Rrs_* variables sorted by wavelength
        rrs_vars = sorted([v for v in group.variables if v.startswith("Rrs_")],
                          key=lambda v: float(v.split("_")[1]))
        native_wls = []
        native_means = []
        for varname in rrs_vars:
            var = group.variables[varname]
            data_raw = var[:]  # may be masked array
            # 1. Convert masked array to float with NaN
            if isinstance(data_raw, np.ma.MaskedArray):
                arr = data_raw.astype(np.float32).filled(np.nan)
            else:
                arr = data_raw.astype(np.float32)
            # 2. Mask fill values
            fill = None
            try:
                fill = var.getncattr("_FillValue")
            except Exception:
                try:
                    fill = var.getncattr("fill_value")
                except Exception:
                    fill = None
            if fill is not None:
                arr[arr == fill] = np.nan
            # 3. Apply scale/offset
            try:
                sf = var.getncattr("scale_factor")
                arr = arr * sf
            except Exception:
                pass
            try:
                off = var.getncattr("add_offset")
                arr = arr + off
            except Exception:
                pass
            # 4. Average over the region
            masked = np.where(region_mask, arr, np.nan)
            mean_val = np.nanmean(masked)
            native_wls.append(float(varname.split("_")[1]))
            native_means.append(mean_val)
        ds.close()

        if not native_wls:
            return np.array([]), np.array([])
        return np.array(native_wls, dtype=float), np.array(native_means, dtype=np.float32)

    except Exception as e:
        print(f"[ERROR] process_granule_modis {path}: {e}")
        return np.array([]), np.array([])

def process_granule_sentinel(path: str, bbox: tuple) -> (np.ndarray, np.ndarray):
    """
    Process a Sentinel-3 granule: returns native wavelengths and mean reflectances.
    """
    try:
        ds = Dataset(path, "r")
        lat = ds.groups["navigation_data"].variables["latitude"][:]
        lon = ds.groups["navigation_data"].variables["longitude"][:]

        lon_min, lat_min, lon_max, lat_max = bbox
        region_mask = (
            (lat >= lat_min) & (lat <= lat_max) &
            (lon >= lon_min) & (lon <= lon_max)
        )
        if not np.any(region_mask):
            ds.close()
            return np.array([]), np.array([])

        gdata = ds.groups["geophysical_data"]
        # Gather Rrs_* variables
        items = []
        for name in gdata.variables:
            if name.startswith("Rrs_"):
                try:
                    wl = float(name.split("_")[1])
                    items.append((wl, name))
                except:
                    pass
        items.sort()
        wls = []
        means = []
        for wl, name in items:
            var = gdata.variables[name]
            data_raw = var[:]
            if isinstance(data_raw, np.ma.MaskedArray):
                arr = data_raw.astype(np.float32).filled(np.nan)
            else:
                arr = data_raw.astype(np.float32)
            # Mask fill
            fill = None
            try:
                fill = var.getncattr("_FillValue")
            except Exception:
                try:
                    fill = var.getncattr("fill_value")
                except Exception:
                    fill = None
            if fill is not None:
                arr[arr == fill] = np.nan
            # Scale/offset
            try:
                sf = var.getncattr("scale_factor")
                arr = arr * sf
            except Exception:
                pass
            try:
                off = var.getncattr("add_offset")
                arr = arr + off
            except Exception:
                pass
            masked = np.where(region_mask, arr, np.nan)
            mean_val = np.nanmean(masked)
            wls.append(wl)
            means.append(mean_val)
        ds.close()

        if not wls:
            return np.array([]), np.array([])
        return np.array(wls, dtype=float), np.array(means, dtype=np.float32)

    except Exception as e:
        print(f"[ERROR] process_granule_sentinel {path}: {e}")
        return np.array([]), np.array([])


def process_granule_pace(
    path: str,
    wave_pace: np.ndarray,
    lat_bins: np.ndarray,
    lon_bins: np.ndarray,
    cache_dir: str,
    bbox: tuple,
    combined_wavelengths: np.ndarray
):
    """
    Extract PACE L2 AOP granule Rrs bands, apply QA mask, and interpolate to combined wavelengths.
    Save results as .npz cache.
    """
    base = os.path.basename(path)
    bbox_str = "_".join(f"{coord:.5f}" for coord in bbox)
    cache_file = os.path.join(cache_dir, f"{base}_{bbox_str}.npz")
    if os.path.exists(cache_file):
        return

    logger.info(f"Processing PACE granule with QA and interpolation: {base}")
    try:
        with xr.open_dataset(path, group="navigation_data") as nav:
            lat = nav["latitude"].values
            lon = nav["longitude"].values

        with xr.open_dataset(path, group="geophysical_data") as ds:
            # QA flags and masks
            if "l2_flags" in ds:
                flags = ds["l2_flags"].values
                flag_masks = ds["l2_flags"].attrs.get("flag_masks", None)
                flag_meanings = ds["l2_flags"].attrs.get("flag_meanings", "")
                if flag_masks is not None:
                    names = flag_meanings.split()
                    mask_map = {name: np.uint32(mask) for name, mask in zip(names, flag_masks)}
                else:
                    mask_map = {}
            else:
                flags = None
                mask_map = {}

            rrs = ds["Rrs"].values  # shape: (bands, lat, lon) or (lat, lon, bands)?
            # Confirm shape: Assume (bands, lat, lon)
            # If (lat, lon, bands), transpose as needed
            # Here let's assume (bands, lat, lon)
            if rrs.shape[0] == len(wave_pace):
                # good
                pass
            elif rrs.shape[-1] == len(wave_pace):
                # transpose to (bands, lat, lon)
                rrs = np.transpose(rrs, (2, 0, 1))
            else:
                raise ValueError("Unexpected shape for Rrs data")

            undesired = [
                "LAND", "CLDICE", "HIGLINT", "MODGLINT",
                "STRAYLIGHT", "ATMFAIL", "ATMWARN", "NAVFAIL", "NAVWARN",
                "SEAICE", "HISOLZEN", "HISATZEN", "COASTZ", "ABSAER",
                "MAXAERITER", "FILTER", "BOWTIEDEL", "HIPOL", "PRODFAIL", "PRODWARN"
            ]

            lat_idx_list = []
            lon_idx_list = []
            ch_idx_list = []
            val_list = []

            # Prepare mask for each pixel based on QA and bbox
            # We will process pixel by pixel to interpolate spectra
            # But that's slow; let's vectorize better:

            # Flatten lat/lon and spectra to 2D: pixels x bands
            shape_orig = rrs.shape
            n_bands, n_lat, n_lon = shape_orig

            lat_flat = lat.flatten()
            lon_flat = lon.flatten()
            rrs_2d = rrs.reshape(n_bands, -1).T  # shape (npixels, n_bands)

            # Build initial mask: finite, bbox, QA
            finite_mask = np.all(np.isfinite(rrs_2d), axis=1)
            bbox_mask = (lat_flat >= bbox[1]) & (lat_flat <= bbox[3]) & (lon_flat >= bbox[0]) & (lon_flat <= bbox[2])

            qa_mask = np.ones_like(finite_mask, dtype=bool)
            if flags is not None and mask_map:
                flags_flat = flags.flatten()
                for name in undesired:
                    if name in mask_map:
                        qa_mask &= (flags_flat & mask_map[name]) == 0

            total_mask = finite_mask & bbox_mask & qa_mask

            if not np.any(total_mask):
                logger.info("No valid pixels after masking for granule %s", base)
                return

            valid_pixels_idx = np.where(total_mask)[0]
            lat_valid = lat_flat[valid_pixels_idx]
            lon_valid = lon_flat[valid_pixels_idx]
            spectra_valid = rrs_2d[valid_pixels_idx, :]  # shape (n_valid, n_bands_native)

            # Interpolate each pixel's spectrum to combined_wavelengths
            # Using scipy interp1d for handling NaNs safely:

            interp_func = interp1d(
                wave_pace,
                spectra_valid.T,
                kind="linear",
                bounds_error=False,
                fill_value=np.nan,
                assume_sorted=True,
            )
            spectra_interp = interp_func(combined_wavelengths).T  # shape (n_valid, n_combined)

            valid_pixel_mask = ~np.all(np.isnan(spectra_interp), axis=1)
            lat_valid = lat_valid[valid_pixel_mask]
            lon_valid = lon_valid[valid_pixel_mask]
            spectra_interp = spectra_interp[valid_pixel_mask, :]

            # Find indices for lat, lon bins
            lat_idx = np.searchsorted(lat_bins, lat_valid) - 1
            lon_idx = np.searchsorted(lon_bins, lon_valid) - 1

            # Store as sparse arrays (pixel-wise)
            for pix_i in range(len(lat_valid)):
                li = lat_idx[pix_i]
                lj = lon_idx[pix_i]
                if li < 0 or lj < 0 or li >= len(lat_bins) - 1 or lj >= len(lon_bins) - 1:
                    continue  # skip out of range

                for ch_i, val in enumerate(spectra_interp[pix_i]):
                    if np.isfinite(val):
                        lat_idx_list.append(li)
                        lon_idx_list.append(lj)
                        ch_idx_list.append(ch_i)
                        val_list.append(val)

        # Save to cache file
        np.savez_compressed(
            cache_file,
            lat_idx=np.array(lat_idx_list, dtype=np.int16),
            lon_idx=np.array(lon_idx_list, dtype=np.int16),
            ch_idx=np.array(ch_idx_list, dtype=np.int16),
            val=np.array(val_list, dtype=np.float32),
        )
        logger.info(f"Saved interpolated PACE cache: {cache_file}")

    except Exception as e:
        logger.error(f"Failed to process PACE granule {base}: {e}")



# ---------------------------
# Load Daily Data per Sensor
# ---------------------------

def load_daily_spectrum(
    date: datetime,
    sensor: str,
    bbox: tuple,
    wave_dict: Dict[str, np.ndarray],
    data_dir: str,
    cache_dir: str,
    combined_wavelengths: np.ndarray
) -> np.ndarray:
    """
    For a given date and sensor, search granules, process each to get native wavelengths/means,
    align by exact match to combined_wavelengths, accumulate sum & count, then return 1D daily spectrum.
    Always returns a numeric np.ndarray of length len(combined_wavelengths) (NaNs if no data).
    """
    date_str = date.strftime("%Y-%m-%d")
    temporal = (date_str, date_str)

    # Choose process function based on sensor
    if sensor in ["MODISA_L2_OC", "MODIST_L2_OC"]:
        short_names = [sensor]
        process_fn = process_granule_modis
    elif sensor in ["OLCIS3A_L2_EFR_OC", "OLCIS3B_L2_EFR_OC"]:
        short_names = [sensor]
        process_fn = process_granule_sentinel
    elif sensor == "PACE_OCI_L2_AOP":
        short_names = ["PACE_OCI_L2_AOP"]
        process_fn = process_granule_pace
    else:
        raise ValueError(f"Unsupported sensor: {sensor}")

    # 1) Search granules for that day
    all_granules = []
    for sn in short_names:
        try:
            results = safe_search(sn, temporal, bbox)
            all_granules.extend(results)
        except Exception as e:
            logger.warning(f"Search failed for {sn} on {date_str}: {e}")
    if not all_granules:
        # No granules → return all-NaN spectrum
        return np.full(len(combined_wavelengths), np.nan, dtype=np.float32)

    # 2) Download missing granules
    to_download = []
    for granule in all_granules:
        granule_id = granule.get("granule_id") or \
            granule["umm"]["DataGranule"]["ArchiveAndDistributionInformation"][0]["Name"]
        base = os.path.basename(granule_id)
        local_nc = os.path.join(data_dir, base)
        if not os.path.exists(local_nc):
            to_download.append(granule)
    if to_download:
        logger.info(f"Downloading {len(to_download)} granules for {sensor} on {date_str}…")
        paths = safe_download(to_download, data_dir)
        if not paths:
            logger.error(f"Failed to download some granules for {sensor} on {date_str}")
            # proceed with whatever local files exist

    # Prepare accumulators for the day
    n_bands = len(combined_wavelengths)
    sum_arr = np.zeros(n_bands, dtype=np.float64)
    count_arr = np.zeros(n_bands, dtype=np.int32)
    tol = 1e-6

    # Ensure cache_dir exists
    os.makedirs(cache_dir, exist_ok=True)
    # Precompute bbox string for cache filenames
    bbox_str = "_".join(f"{coord:.5f}" for coord in bbox)

    for granule in all_granules:
        granule_id = granule.get("granule_id") or \
            granule["umm"]["DataGranule"]["ArchiveAndDistributionInformation"][0]["Name"]
        base = os.path.basename(granule_id)
        local_nc = os.path.join(data_dir, base)
        if not os.path.exists(local_nc):
            logger.warning(f"Granule file missing locally: {local_nc}; skipping")
            continue

        # Cache filename for this granule+bbox
        cache_file = os.path.join(cache_dir, f"{sensor}_{base}_{bbox_str}.npy")

        # Attempt to load cached aligned spectrum
        spec_aligned = None
        if os.path.exists(cache_file):
            try:
                spec_loaded = np.load(cache_file)
                # verify shape
                if isinstance(spec_loaded, np.ndarray) and spec_loaded.shape == (n_bands,):
                    spec_aligned = spec_loaded.astype(np.float32)
                else:
                    logger.debug(f"Cache file {cache_file} has wrong shape; will re-compute")
            except Exception:
                logger.debug(f"Failed to load cache {cache_file}; will re-compute")

        if spec_aligned is None:
            # Process granule to get native wavelengths and means
            native_wls, native_means = process_fn(local_nc, bbox)
            if native_wls is None or native_means is None:
                # Treat as no data
                logger.debug(f"{sensor} process_granule returned None for {local_nc}; skipping")
                continue
            if native_wls.size == 0 or native_means.size == 0:
                # No valid bands in this granule
                logger.debug(f"No valid native bands for {local_nc}; skipping")
                continue

            # Build aligned spectrum array of length n_bands, init NaN
            spec_aligned = np.full(n_bands, np.nan, dtype=np.float32)
            # Align exact matches
            for wl, mean in zip(native_wls, native_means):
                if not np.isfinite(mean):
                    continue
                idx = np.where(np.isclose(combined_wavelengths, wl, atol=tol))[0]
                if idx.size > 0:
                    spec_aligned[idx[0]] = mean
            # Save to cache
            try:
                np.save(cache_file, spec_aligned)
            except Exception as e:
                logger.debug(f"Failed to save cache {cache_file}: {e}")

        # Accumulate this granule’s spec_aligned into sum/count
        finite_mask = np.isfinite(spec_aligned)
        if np.any(finite_mask):
            sum_arr[finite_mask] += spec_aligned[finite_mask]
            count_arr[finite_mask] += 1
        else:
            logger.debug(f"{local_nc}: aligned spectrum all NaN; skipping accumulation")

    # After looping all granules, compute daily average
    with np.errstate(divide='ignore', invalid='ignore'):
        daily = sum_arr / count_arr
    # Wherever count == 0, set NaN
    daily[count_arr == 0] = np.nan
    # Convert to float32
    daily = daily.astype(np.float32)

    return daily

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

def aggregate_sensors_simple(
    start_date: str,
    end_date: str,
    bbox: Tuple[float, float, float, float],
    sensors: List[str],
    data_dir: str = "../Data/",
    cache_dir: str = "../Cache/",
    wave_dict: Dict[str, np.ndarray] = None,
    combined_wavelengths: np.ndarray = None
) -> Tuple[np.ndarray, Dict]:
    """
    For each day in [start_date, end_date], for each sensor compute daily spectrum via load_daily_spectrum,
    then average across sensors. Returns arr: shape (n_days, n_bands) and meta with 'wavelengths' and 'dates'.
    """
    # Parse dates
    sd = datetime.strptime(start_date, "%Y-%m-%d")
    ed = datetime.strptime(end_date, "%Y-%m-%d")
    date_objs = [sd + timedelta(days=i) for i in range((ed - sd).days + 1)]
    date_list_str = [d.strftime("%Y-%m-%d") for d in date_objs]
    total_days = len(date_objs)

    # Prepare wave_dict if None
    if wave_dict is None:
        wave_dict = {}
        for sensor in sensors:
            wave = get_reference_wavelengths(sensor, bbox)
            wave = wave[wave <= 2300]
            wave_dict[sensor] = wave

    # Combined wavelengths
    if combined_wavelengths is None:
        all_waves = np.unique(np.concatenate([wave_dict[s] for s in sensors]))
        combined_wavelengths = np.sort(all_waves)
    n_bands = len(combined_wavelengths)

    # Output array: (n_days, n_bands)
    arr = np.full((total_days, n_bands), np.nan, dtype=np.float32)

    # Ensure data_dir/cache_dir exist
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    for idx, date in enumerate(date_objs):
        date_str = date_list_str[idx]
        logger.info(f"Aggregating date {date_str} ({idx+1}/{total_days})")
        per_sensor_specs = []
        for sensor in sensors:
            try:
                daily_spec = load_daily_spectrum(
                    date=date,
                    sensor=sensor,
                    bbox=bbox,
                    wave_dict=wave_dict,
                    data_dir=data_dir,
                    cache_dir=cache_dir,
                    combined_wavelengths=combined_wavelengths
                )
                if daily_spec is None:
                    # replace with all-NaN array of correct shape
                    daily_spec = np.full(len(combined_wavelengths), np.nan, dtype=np.float32)
                if np.all(np.isnan(daily_spec)):
                    logger.info(f"Sensor {sensor} returned all-NaN on {date_str}")
                else:
                    per_sensor_specs.append(daily_spec)
            except Exception as e:
                logger.warning(f"Error loading daily spectrum for {sensor} on {date_str}: {e}")
        if per_sensor_specs:
            stacked = np.stack(per_sensor_specs, axis=0)
            arr[idx] = np.nanmean(stacked, axis=0)
        else:
            logger.warning(f"No valid sensor data on {date_str}; leaving NaNs")
    meta = {
        "wavelengths": combined_wavelengths,
        "dates": date_list_str
    }
    return arr, meta


def extract_mean_spectrum(
    lat: float,
    lon: float,
    target_date: pd.Timestamp,
    sensors: list,
    bbox_size_km: float,
    pixel_count: int,
    data_dir: str,
    cache_dir: str,
    wave_dict: Dict[str, np.ndarray],
    aggregate_fn,  # reference to aggregate_sensors
):
    """
    For a single field observation at (lat, lon) and target_date, 
    build a small bbox of size pixel_count × pixel_count at approx bbox_size_km per pixel,
    call aggregate_sensors over ±4 days around target_date, and return a 1D mean spectrum
    
    Returns:
      spectrum: 1D np.ndarray of length n_bands (with np.nan where no data)
      wavelengths: 1D np.ndarray of length n_bands
    """
    # 1. Build bbox in degrees
    res_lat = bbox_size_km / 111.0
    res_lon = bbox_size_km / (111.0 * math.cos(math.radians(lat)))
    half = pixel_count // 2
    delta_lat = half * res_lat
    delta_lon = half * res_lon
    bbox = (lon - delta_lon, lat - delta_lat, lon + delta_lon, lat + delta_lat)

    # 2. Time window ±4 days
    start_date = target_date - pd.Timedelta(days=4)
    end_date   = target_date + pd.Timedelta(days=4)

    # 3. Call aggregate_sensors
    arr4d, meta = aggregate_fn(
        start_date=start_date,
        end_date=end_date,
        bbox=bbox,
        sensors=sensors,
        resolution=(res_lat, res_lon),
        data_dir=data_dir,
        cache_dir=cache_dir,
        wave_dict=wave_dict
    )
    dates = meta.get("dates", meta.get("date_list", None))
    # Depending on meta key: adjust if needed
    # If date strings, convert to pd.Timestamp
    if dates and not isinstance(dates[0], pd.Timestamp):
        dates = [pd.to_datetime(d) for d in dates]
    wavelengths = meta["wavelengths"]  # 1D array

    # 4. Compute exponential weights over days
    weights = np.array([np.exp(-abs((d - target_date).days)) for d in dates], dtype=float)
    total = weights.sum()
    if total <= 0:
        # no valid weights
        return None, wavelengths
    weights /= total

    # 5. Accumulate weighted mean per band over spatial pixels and days
    n_days, h, w, n_bands = arr4d.shape
    weighted_sum = np.zeros(n_bands, dtype=float)
    weight_total = np.zeros(n_bands, dtype=float)

    for i in range(n_days):
        data_i = arr4d[i]  # shape (h, w, n_bands)
        w_i = weights[i]
        # reshape spatial dims
        flat = data_i.reshape(-1, n_bands)  # (h*w, n_bands)
        finite = np.isfinite(flat)  # mask
        # sum & count per band
        sum_b = np.nansum(np.where(finite, flat, 0.0), axis=0)
        count_b = finite.sum(axis=0).astype(float)
        valid = count_b > 0
        if np.any(valid):
            weighted_sum[valid] += w_i * (sum_b[valid] / count_b[valid])
            weight_total[valid] += w_i

    # 6. Build final spectrum
    spectrum = np.full(n_bands, np.nan, dtype=np.float32)
    valid_bands = weight_total > 0
    spectrum[valid_bands] = weighted_sum[valid_bands].astype(np.float32)
    return spectrum, wavelengths