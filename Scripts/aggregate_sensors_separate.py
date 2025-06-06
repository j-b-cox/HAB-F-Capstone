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

def load_daily_sensor(date: datetime,
                      sensor: str,
                      bbox: Tuple[float, float, float, float],
                      lat_bins: np.ndarray,
                      lon_bins: np.ndarray,
                      wave_dict: Dict[str, np.ndarray],
                      cache_dir: str) -> Tuple[np.ndarray, np.ndarray]:
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
    all_granules = []

    # Determine which collections to search
    if sensor in ["MODISA_L2_OC", "MODIST_L2_OC"]:
        short_names = [sensor]  # <-- only use the current sensor
        wave_ref = wave_dict[sensor]  # <-- use per-sensor wavelengths
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

    # Search each relevant collection
    for sn in short_names:
        results = safe_search(sn, temporal, bbox)
        all_granules.extend(results)

    nlat = len(lat_bins) - 1
    nlon = len(lon_bins) - 1
    n_bands = len(wave_ref)

    if not all_granules:
        logger.warning(f"No granules for {sensor} on {date_str}. Returning NaNs.")
        return np.full((nlat, nlon, n_bands), np.nan, dtype=np.float32), wave_ref

    # Download any missing granules
    to_download = []
    for granule in all_granules:
        granule_id = granule.get("granule_id") or \
                     granule["umm"]["DataGranule"]["ArchiveAndDistributionInformation"][0]["Name"]
        base = os.path.basename(granule_id)
        local_nc = os.path.join("../Data/", base)
        cache_file = os.path.join(cache_dir, f"{base}.npz")
        if not os.path.exists(cache_file):
            to_download.append(granule)

    if to_download:
        logger.info(f"Downloading {len(to_download)} new granules for {sensor} on {date_str}...")
        paths = safe_download(to_download, "../Data/")
        if not paths:
            logger.error(f"Failed to download granules for {sensor} on {date_str}. Returning NaNs.")
            return np.full((nlat, nlon, n_bands), np.nan, dtype=np.float32), wave_ref
    else:
        logger.info(f"All granules for {sensor} on {date_str} already cached.")

    # Build list of granule file paths
    granule_paths = []
    for granule in all_granules:
        granule_id = granule.get("granule_id") or \
                     granule["umm"]["DataGranule"]["ArchiveAndDistributionInformation"][0]["Name"]
        base = os.path.basename(granule_id)
        local_nc = os.path.join("../Data/", base)
        granule_paths.append(local_nc)

    # Initialize accumulators
    sum_all = np.zeros((n_bands, nlat, nlon), dtype=np.float64)
    count_all = np.zeros((n_bands, nlat, nlon), dtype=np.int32)

    # Process each granule
    for path in granule_paths:
        base = os.path.basename(path)
        cache_file = os.path.join(cache_dir, f"{base}.npz")

        if not os.path.exists(cache_file):
            process_fn(path, wave_ref, lat_bins, lon_bins, cache_dir, bbox)

        if not os.path.exists(cache_file):
            logger.error(f"Cache missing for granule {base}. Skipping.")
            continue

        print(f"Checking cache file: {cache_file}, exists: {os.path.exists(cache_file)}")
        data = np.load(cache_file)
        lat_idx = data["lat_idx"]
        lon_idx = data["lon_idx"]
        ch_idx = data["ch_idx"]
        val = data["val"]

        for j in range(len(val)):
            li = lat_idx[j]
            lj = lon_idx[j]
            ci = ch_idx[j]
            if 0 <= li < nlat and 0 <= lj < nlon and 0 <= ci < n_bands:
                sum_all[ci, li, lj] += val[j]
                count_all[ci, li, lj] += 1

    # Compute daily average
    with np.errstate(invalid="ignore", divide="ignore"):
        daily_avg = sum_all / count_all
        daily_avg[count_all == 0] = np.nan
    print(f"{sensor} {date_str}: sum_all has nonzero: {np.sum(sum_all != 0)}")
    print(f"{sensor} {date_str}: count_all max: {np.max(count_all)}")
    print(f"{sensor} {date_str}: daily_avg NaNs: {np.isnan(daily_avg).sum()} / {daily_avg.size}")


    # Transpose to (nlat, nlon, n_bands)
    daily_avg = np.transpose(daily_avg, (1, 2, 0)).astype(np.float32)
    return daily_avg, wave_ref

def aggregate_sensors_separate(start_date: str,
                               end_date: str,
                               bbox: Tuple[float, float, float, float],
                               sensors: List[str],
                               resolution: float = 0.01,
                               data_dir: str = "../Data/",
                               cache_dir: str = "../Cache/") -> Tuple[np.ndarray, Dict]:
    """
    Aggregate daily Rrs from multiple sensors over the given date range and bbox,
    **without averaging across sensors**.
    Output shape: (n_days, nlat, nlon, n_sensors, max_n_bands)
    Sensor data with fewer bands are padded with NaNs on last axis.

    Returns:
        ndarray_all: 5D array (n_days, nlat, nlon, n_sensors, max_n_bands)
        metadata: dict with keys
          - 'wavelengths': list of 1D arrays, one per sensor
          - 'lat': lat centers
          - 'lon': lon centers
          - 'date_list': list of date strings
          - 'sensors': list of sensor names in order of 4th dimension
    """
    logger.info("Authenticating Earthdata...")
    auth = earthaccess.login(persist=True)

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    lat_bins, lon_bins, lat_centers, lon_centers = build_grid(bbox, resolution)
    nlat, nlon = len(lat_centers), len(lon_centers)

    # Retrieve wavelengths per sensor
    wave_dict = {}
    sensor_wave_list = []

    for sensor in sensors:
        wave = get_reference_wavelengths(sensor, bbox)
        wave = wave[wave <= 2300]
        wave_dict[sensor] = wave
        print(f"{sensor} wavelengths: {wave}")
        sensor_wave_list.append(wave)

    # Determine max bands across all sensors for padding)
    max_n_bands = max(len(wave_dict[w]) for w in wave_dict)
    logger.info(f"max_n_bands: {max_n_bands}")
    n_sensors = len(sensors)

    date_objs = parse_date_range(start_date, end_date)
    total_days = len(date_objs)
    date_list_str = [d.strftime("%Y-%m-%d") for d in date_objs]
    logger.info(f"Aggregating {total_days} days from {start_date} to {end_date}.")

    # Initialize 5D output array with NaNs
    ndarray_all = np.full((total_days, nlat, nlon, n_sensors, max_n_bands), np.nan, dtype=np.float32)

    def normalize_hybrid_per_sensor(arr: np.ndarray) -> np.ndarray:
        """
        Safely apply hybrid normalization per spectrum:
        - Normalize shape (unit vector of spectrum)
        - Multiply by global median magnitude
        """
        nlat, nlon, nbands = arr.shape
        shape = np.full_like(arr, np.nan, dtype=np.float32)
        norms = []

        for i in range(nlat):
            for j in range(nlon):
                spectrum = arr[i, j, :]
                valid = ~np.isnan(spectrum)
                if np.sum(valid) >= 2:
                    vec = spectrum[valid]
                    norm = np.linalg.norm(vec)
                    if norm > 0:
                        shape_vec = vec / norm
                        shape[i, j, valid] = shape_vec
                        norms.append(norm)

        if not norms:
            print("⚠️ No valid spectra found for normalization.")
            return shape  # All NaN

        magnitude = np.median(norms)
        return shape * magnitude



    # In your aggregation function loop:
    for day_idx, current_date in enumerate(tqdm(date_objs, desc="Processing dates")):
        date_str = current_date.strftime("%Y-%m-%d")
        logger.info(f"Processing {date_str} ({day_idx+1}/{total_days})")

        for sensor_idx, sensor in enumerate(sensors):
            logger.info(f"Processing sensor {sensor}")
            daily_arr, wave_ref = load_daily_sensor(
                date=current_date,
                sensor=sensor,
                bbox=bbox,
                lat_bins=lat_bins,
                lon_bins=lon_bins,
                wave_dict=wave_dict,
                cache_dir=cache_dir
            )  # daily_arr shape: (nlat, nlon, n_bands_sensor)

            n_bands_sensor = daily_arr.shape[2]

            # 🆕 Hybrid normalization step
            
            #daily_arr = normalize_hybrid_per_sensor(daily_arr)
            print(f"{sensor} {date_str}: daily_arr non-NaNs = {np.sum(~np.isnan(daily_arr))} / {daily_arr.size}")

            # Place sensor data into ndarray with padding if needed
            logger.info(f"daily_arr.shape: {daily_arr.shape}")
            logger.info(f"n_bands_sensor: {n_bands_sensor}")
            logger.info(f"ndarray_all.shape: {ndarray_all.shape}")
            assert daily_arr.shape[0] == ndarray_all.shape[1], f"Lat mismatch: {daily_arr.shape[0]} vs {ndarray_all.shape[1]}"
            assert daily_arr.shape[1] == ndarray_all.shape[2], f"Lon mismatch: {daily_arr.shape[1]} vs {ndarray_all.shape[2]}"   
            assert daily_arr.shape[2] <= ndarray_all.shape[4], f"Bands mismatch: {daily_arr.shape[2]} vs {ndarray_all.shape[4]}"
            print(f"Assigning day {day_idx}, sensor {sensor_idx}: daily_arr shape={daily_arr.shape}")
            ndarray_all[day_idx, :, :, sensor_idx, :n_bands_sensor] = daily_arr
            print(f"Assigned slice shape: {ndarray_all[day_idx, :, :, sensor_idx, :n_bands_sensor].shape}")
            print(f"Post-assignment nan count: {np.isnan(ndarray_all[day_idx, :, :, sensor_idx, :n_bands_sensor]).sum()}")

    metadata = {
        "wavelengths": [wave_dict[s] for s in sensors],
        "sensors": sensors,
        "lat": lat_centers,
        "lon": lon_centers,
        "date_list": date_list_str
    }

    return ndarray_all, metadata

if __name__ == "__main__":
    START_DATE = "2024-04-14"
    END_DATE   = "2024-04-20"
    BBOX       = (-83.62, 41.34, -82, 42.27)
    SENSORS    = ["MODISA_L2_OC", "MODIST_L2_OC", "OLCIS3A_L2_EFR_OC", "OLCIS3B_L2_EFR_OC", "PACE_OCI_L2_AOP"]
    RES        = 0.01

    arr_5d, meta = aggregate_sensors_separate(
        start_date=START_DATE,
        end_date=END_DATE,
        bbox=BBOX,
        sensors=SENSORS,
        resolution=RES
    )
    # Average across sensors for each pixel and band
    ndarray_avg = np.nanmean(arr_5d, axis=3)  # shape: (ndays, nlat, nlon, max_n_bands)


    # Save output and metadata
    date_range_str = f"{START_DATE.replace('-', '')}-{END_DATE.replace('-', '')}"
    np.save(f"../Data/aggregated_separate_{date_range_str}.npy", arr_5d)
    np.save(f"../Data/aggregated_avg_{date_range_str}.npy", ndarray_avg)
    with open(f"../Data/aggregated_separate_{date_range_str}_metadata.pkl", "wb") as f:
        pickle.dump(meta, f)
    logger.info(f"Saved separate sensor 5D array and metadata.")
