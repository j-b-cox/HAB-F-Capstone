import os
import re
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from datetime import datetime, timedelta
import xarray as xr
from pyresample import geometry, kd_tree
import earthaccess

def extract_datetime_from_filename(path):
    """
    Extract datetime from filename by searching for YYYYMMDDThhmmss.
    Returns a datetime or None if not found/parsable.
    """
    filename = os.path.basename(path)
    m = re.search(r"(\d{8}T\d{6})", filename)
    if not m:
        print(f"[WARN] No timestamp pattern in filename: {filename}")
        return None
    ts = m.group(1)  # e.g. "20250615T102030"
    try:
        return datetime.strptime(ts, "%Y%m%dT%H%M%S")
    except Exception as e:
        print(f"[ERROR] Failed to parse timestamp '{ts}' in {filename}: {e}")
        return None


def standardize_patch(patch, target_h=4, target_w=4, target_c=10):
    """
    Resize a patch array of shape (h0, w0, c0) to (target_h, target_w, target_c).
    Spatial resizing uses nearest-neighbor sampling.
    Channel resizing:
      - If c0 > target_c: select uniformly spaced indices to reduce to target_c.
      - If c0 < target_c: pad extra channels with NaN.
    """
    h0, w0, c0 = patch.shape
    
    # Spatial resize via nearest neighbor sampling
    ys = np.linspace(0, h0 - 1, target_h)
    xs = np.linspace(0, w0 - 1, target_w)
    iy = np.round(ys).astype(int)
    ix = np.round(xs).astype(int)
    # shape (target_h, target_w, c0)
    resized = patch[iy[:, None], ix[None, :], :]
    
    # Channel resize or pad
    if c0 == target_c:
        result = resized
    elif c0 > target_c:
        # Pick uniformly spaced channel indices
        idx_c = np.round(np.linspace(0, c0 - 1, target_c)).astype(int)
        result = resized[:, :, idx_c]
    else:
        # Pad with NaN
        result = np.full((target_h, target_w, target_c), np.nan, dtype=patch.dtype)
        result[:, :, :c0] = resized
    
    return result

def process_all_rows():
    # Constants
    PLOT_BBOX = (-83.5, 41.3, -82.45, 42.2)
    BOX_HALF_DEG = 0.0225  # ~5km
    TARGET_RESOLUTION = 0.005  # ~1km grid
    PROGRESS_SAVE_INTERVAL = 1  # save every N rows
    SENSOR_SHORT_NAME = "OLCIS3A_L2_EFR_OC"
    TRAINING_FILE = f"./training_data_{SENSOR_SHORT_NAME}.npy"
    
    lon_min, lat_min, lon_max, lat_max = PLOT_BBOX
    margin = 0.1
    target_lons = np.arange(lon_min - margin, lon_max + margin, TARGET_RESOLUTION)
    target_lats = np.arange(lat_min - margin, lat_max + margin, TARGET_RESOLUTION)
    lon2d, lat2d = np.meshgrid(target_lons, target_lats)
    area_def = geometry.GridDefinition(lons=lon2d, lats=lat2d)

    # Authenticate
    auth = earthaccess.login(persist=True)

    # Load station data
    df = pd.read_csv("glrl-hab-data.csv", index_col=0)

    # Prepare storage
    if os.path.exists(TRAINING_FILE):
        print(f"[INFO] Loading existing training data from {TRAINING_FILE}")
        existing = np.load(TRAINING_FILE, allow_pickle=True)
        # Expect each entry as tuple: (row_index, labels_array, patch_flat_array)
        processed_indices = set(entry[0] for entry in existing)
        results = existing.tolist()
    else:
        processed_indices = set()
        results = []

    def extract_datetime_from_modis_filename(path):
        filename = os.path.basename(path)
        try:
            timestamp_str = filename.split(".")[1]  # e.g., "20120514T173000"
            return datetime.strptime(timestamp_str, "%Y%m%dT%H%M%S")
        except Exception as e:
            print(f"[ERROR] Failed to extract datetime from {filename}: {e}")
            return None


    completed_file = f"completed_rows_{SENSOR_SHORT_NAME}.txt"
    completed_idx = set()
    if os.path.exists(completed_file):
        with open(completed_file, "r") as f:
            for line in f:
                line = line.strip()
                if line.isdigit():
                    completed_idx.add(int(line))

    # Prepare storage for valid patches
    results = []
    processed_indices = set(completed_idx)

    total_rows = len(df)
    for count, (i, row) in enumerate(df.iterrows()):
        if i in processed_indices:
            print(f"[INFO] Skipping row index {i} (already processed)")
            continue

        station_lat = row['lat']
        station_lon = row['lon']
        timestamp = pd.to_datetime(row['timestamp'])
        print(f"\n[INFO] ({count}/{total_rows}) Processing row index {i} | Station: {row.get('station_name', '')} | {timestamp}")

        if SENSOR_SHORT_NAME in ["OLCIS3A_L2_EFR_OC", "OLCIS3A_L2_EFR_OC"] and timestamp < pd.to_datetime("2018-02-01"):
            continue

        mini_bbox = (
            station_lon - BOX_HALF_DEG,
            station_lat - BOX_HALF_DEG,
            station_lon + BOX_HALF_DEG,
            station_lat + BOX_HALF_DEG
        )

        start = (timestamp - timedelta(hours=48)).strftime('%Y-%m-%dT%H:%M:%SZ')
        end   = (timestamp + timedelta(hours=48)).strftime('%Y-%m-%dT%H:%M:%SZ')

        print(f"[INFO] Searching for granules in time range {start} to {end} within 5km box.")
        try:
            results_search = earthaccess.search_data(
            short_name=SENSOR_SHORT_NAME,
            temporal=(start, end),
            bounding_box=mini_bbox
        )
        except Exception as e:
            print(f"[ERROR] Search failed for row {i}: {e}")
            continue

        if not results_search:
            print(f"[WARN] No granules found for station {row.get('station_name', '')}")
            continue

        try:
            paths = earthaccess.download(results_search, "granules")
        except Exception as e:
            print(f"[ERROR] Download failed for row {i}: {e}")
            continue
        print(f"[INFO] {len(paths)} granules downloaded.")

        # Sort by closeness to timestamp
        sorted_paths = sorted(paths, key=lambda p: abs((extract_datetime_from_modis_filename(p) or timestamp) - timestamp))

        found_valid = False
        patch_array = None
        for best_file in sorted_paths:
            print(f"[INFO] Trying granule: {os.path.basename(best_file)}")
            try:
                with xr.open_dataset(best_file, group="geophysical_data") as obs, \
                    xr.open_dataset(best_file, group="navigation_data") as nav:
                    nav = nav.set_coords(("longitude","latitude"))
                    dataset = xr.merge((obs, nav.coords))
            except Exception as e:
                print(f"[ERROR] Failed to load {best_file}: {e}")
                continue

            # Prepare target grid
            lon_min, lat_min, lon_max, lat_max = PLOT_BBOX
            margin = 0.1  # or less
            target_lons = np.arange(lon_min - margin, lon_max + margin, TARGET_RESOLUTION)
            target_lats = np.arange(lat_min - margin, lat_max + margin, TARGET_RESOLUTION)
            lon2d, lat2d = np.meshgrid(target_lons, target_lats)
            area_def = geometry.GridDefinition(lons=lon2d, lats=lat2d)

            lons = nav["longitude"].values.flatten()
            lats = nav["latitude"].values.flatten()

            # Identify all Rrs_ bands in dataset
            bands = [name for name in dataset.data_vars if name.startswith("Rrs_")]
            if not bands:
                print(f"[WARN] No Rrs_ bands found in {best_file}")
                continue

            regridded_full = []
            skip_file = False

            for band_name in bands:
                print(f"[INFO] Reprojecting band: {band_name}")

                data = dataset[band_name].values.flatten()

                # Mask lat/lon/data to local area first
                in_bounds = (
                    (lons >= station_lon - 1.0) & (lons <= station_lon + 1.0) &
                    (lats >= station_lat - 1.0) & (lats <= station_lat + 1.0)
                )

                data_local = data[in_bounds]
                lons_local = lons[in_bounds]
                lats_local = lats[in_bounds]

                valid = ~np.isnan(data_local) & ~np.isnan(lons_local) & ~np.isnan(lats_local)

                if not np.any(valid):
                    print(f"[WARN] {band_name} has no valid data near station.")
                    skip_file = True
                    break

                swath_def = geometry.SwathDefinition(lons=lons_local[valid], lats=lats_local[valid])
                result = kd_tree.resample_nearest(
                    swath_def, data_local[valid], area_def,
                    radius_of_influence=5000,
                    fill_value=np.nan
                )

                if not np.any(valid):
                    print(f"[WARN] {band_name} has no valid data.")
                    skip_file = True
                    break

                full_da = xr.DataArray(
                    result,
                    dims=("latitude", "longitude"),
                    coords={"latitude": target_lats, "longitude": target_lons},
                    name=band_name
                ).sel(
                    longitude=slice(PLOT_BBOX[0], PLOT_BBOX[2]),
                    latitude=slice(PLOT_BBOX[1], PLOT_BBOX[3])
                )

                # Check valid percentage in 5km box
                test_da = full_da.sel(
                    longitude=slice(station_lon - BOX_HALF_DEG, station_lon + BOX_HALF_DEG),
                    latitude=slice(station_lat - BOX_HALF_DEG, station_lat + BOX_HALF_DEG)
                )

                if test_da.size == 0:
                    print(f"[WARN] test_da is empty for {band_name}. Skipping.")
                    skip_file = True
                    break

                valid_pct = test_da.count().item() / test_da.size * 100
                print(f"[INFO] {band_name} valid pixel % in 5km box: {valid_pct:.2f}%")

                if valid_pct < 2:
                    print(f"[WARN] Insufficient data in 5km box for {band_name}")
                    skip_file = True
                    break


                regridded_full.append(full_da)

            if skip_file:
                continue
            if len(regridded_full) != len(bands):
                print(f"[ERROR] Not all bands available for a valid patch.")
                continue

            # Extract patch arrays for each band
            patch_list = []
            for da in regridded_full:
                da_patch = da.sel(
                    longitude=slice(station_lon - BOX_HALF_DEG, station_lon + BOX_HALF_DEG),
                    latitude=slice(station_lat - BOX_HALF_DEG, station_lat + BOX_HALF_DEG)
                )
                patch_list.append(da_patch.values)
            # Stack into array shape (h, w, c)
            # After stacking:
            patch_stack = np.stack(patch_list, axis=-1)  # shape (h0, w0, c0)
            print(f"[INFO] Extracted patch shape before standardizing: {patch_stack.shape}")

            # Standardize to 4x4x10
            patch_std = standardize_patch(patch_stack, target_h=4, target_w=4, target_c=10)
            print(f"[INFO] Patch shape after standardizing: {patch_std.shape}")

            # Flatten for saving
            patch_array = patch_std.flatten()

            found_valid = True
            break

        if not found_valid:
            print(f"[WARN] All granules failed for row index {i}.")
            continue

        # Extract 4 target columns
        labels = np.array([
            row.get("particulate_microcystin", np.nan),
            row.get("dissolved_microcystin", np.nan),
            row.get("extracted_phycocyanin", np.nan),
            row.get("extracted_chla", np.nan)
        ], dtype=float)

        # Append tuple (row_index, labels_array, patch_flat_array)
        results.append((i, labels, patch_array))

        # Mark row as processed
        with open(completed_file, "a") as f:
            f.write(f"{i}\n")
        processed_indices.add(i)

        # Periodic save
        if len(results) % PROGRESS_SAVE_INTERVAL == 0:
            np.save(TRAINING_FILE, np.array(results, dtype=object))
            print(f"[INFO] Progress saved: {len(results)} entries so far.")

    # Final save
    np.save(TRAINING_FILE, np.array(results, dtype=object))
    print(f"[INFO] Completed. Total patches saved: {len(results)}")

if __name__ == "__main__":
    # Optionally set start method
    import multiprocessing as mp
    try:
        mp.set_start_method('fork')
    except RuntimeError:
        pass

    process_all_rows()