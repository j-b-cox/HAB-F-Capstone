import os
import math
import logging
import random
import calendar
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import xarray as xr
import earthaccess

from helpers import *
# Make sure process_pace_granule and extract_pace_patch are importable:
# e.g., from your module where you defined them:
# from data_collection_helpers import process_pace_granule, extract_pace_patch

def balance_dataset_by_granule(sensor, max_granule_attempts=137, seed=42):
    """
    Balance by iterating winter-month granules first.
    Handles PACE specially by using process_pace_granule and extract_pace_patch.
    """
    logging.basicConfig(level=logging.INFO)
    logging.info(f"Starting granule-first balancing for sensor={sensor}")
    # Load station DataFrame
    df = pd.read_csv("glrl-hab-data.csv", index_col=0)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    # Load existing results
    results_path = f"training_data_{sensor}.npy"
    if not os.path.exists(results_path):
        logging.error(f"No existing training data at {results_path}; run process_all_granules first.")
        return
    arr = np.load(results_path, allow_pickle=True)
    results = arr.tolist()  # list of tuples: (filename, station_name, label_tuple, patch_flat)
    # Load processed granules set
    processed_txt = f"processed_granules_{sensor}.txt"
    processed_set = set()
    if os.path.exists(processed_txt):
        with open(processed_txt, "r") as f:
            for line in f:
                fn = line.strip()
                if fn:
                    processed_set.add(fn)
    invalid_txt = f"invalid_granules_{sensor}.txt"
    invalid_set = set()
    if os.path.exists(invalid_txt):
        with open(invalid_txt, "r") as f:
            for line in f:
                fn = line.strip()
                if fn:
                    invalid_set.add(fn)

    # Sensor parameters: must match main
    sensor = sensor.upper()
    sensor_map = {
        "MODIS": {
            "short_names": ["MODISA_L2_OC", "MODIST_L2_OC"],
            "res_km": 1.0,
            "pixel_count": 3,
        },
        "SENTINEL": {
            "short_names": ["OLCIS3A_L2_EFR_OC", "OLCIS3A_L2_EFR_OC_NRT",
                            "OLCIS3B_L2_EFR_OC", "OLCIS3B_L2_EFR_OC_NRT"],
            "res_km": 0.4,
            "pixel_count": 8,
        },
        "PACE": {
            "short_names": ["PACE_OCI_L2_AOP", "PACE_OCI_L2_AOP_NRT"],
            "res_km": 1.2,
            "pixel_count": 1,
        }
    }
    if sensor not in sensor_map:
        logging.error(f"Unsupported sensor '{sensor}'")
        return
    params = sensor_map[sensor]
    bbox = (-83.5, 41.3, -82.45, 42.2)
    # Prepare random
    rand = random.Random(seed)

    # If PACE: retrieve wave_all once
    if sensor == "PACE":
        logging.info("Retrieving PACE wavelength list for balancing...")
        # similar to process_all_granules reference retrieval:
        try:
            search_ref = earthaccess.search_data(
                short_name="PACE_OCI_L2_AOP",
                temporal=("2024-06-01", "2024-06-05"),
                bounding_box=bbox,
            )
            if not search_ref:
                raise RuntimeError("No reference files found to retrieve wavelengths.")
            ref_file = earthaccess.download(search_ref, "./Data/")[0]
            wave_all = xr.open_dataset(ref_file, group="sensor_band_parameters")["wavelength_3d"].data
        except Exception as e:
            logging.error(f"Failed to retrieve PACE wavelength list: {e}")
            return
    else:
        wave_all = None

    # Build station-year grouping of existing results
    station_year_records = {}  # (station, year) -> list of mean_micro values
    for rec in results:
        _, station, label_tuple, _ = rec
        t0 = label_tuple[1]
        # Determine year
        if isinstance(t0, pd.Timestamp):
            year = t0.year
        elif isinstance(t0, datetime):
            year = t0.year
        else:
            try:
                dt0 = pd.to_datetime(t0, utc=True)
                year = dt0.year
            except:
                continue
        mean_micro = label_tuple[4]
        key = (station, year)
        station_year_records.setdefault(key, []).append(mean_micro)

    # Compute average lat-lon per station-year
    station_year_loc = {}
    for (station, year) in station_year_records.keys():
        mask = (df["station_name"] == station) & (df["timestamp"].dt.year == year)
        df_st_year = df.loc[mask]
        if df_st_year.empty:
            logging.info(f"No station observations for {station} in year {year}; skipping.")
            continue
        avg_lat = df_st_year["lat"].mean()
        avg_lon = df_st_year["lon"].mean()
        station_year_loc[(station, year)] = (float(avg_lat), float(avg_lon))

    # Build list of station-year needing balancing: pos > neg
    imbalance = {}
    for (station, year), vals in station_year_records.items():
        pos = sum(1 for v in vals if v >= 0.1)
        neg = sum(1 for v in vals if v < 0.1)
        if pos > neg and (station, year) in station_year_loc:
            imbalance[(station, year)] = {"pos": pos, "neg": neg}
    if not imbalance:
        logging.info("No station-years need balancing.")
        return

    # EarthAccess login
    auth = earthaccess.login(persist=True)

    # Helper: list winter months for a year
    def winter_month_ranges(year):
        ranges = []
        for month in [1, 2, 3, 12]:
            try:
                if month == 11:
                    start = datetime(year, month, 15)
                else:
                    start = datetime(year, month, 1)
            except ValueError:
                continue
            if month == 12:
                next_month = datetime(year+1, 1, 1)
            else:
                next_month = datetime(year, month+1, 1)
            ranges.append((start, next_month))
        return ranges

    # Temp download folder
    temp_root = os.path.join("data", sensor, "balance_temp")

    # Build per-year list of stations needing balancing
    year_to_stations = {}
    for (station, year), counts in imbalance.items():
        year_to_stations.setdefault(year, []).append(station)
    years_sorted = sorted(year_to_stations.keys(),
                          key=lambda y: sum(imbalance[(s,y)]["pos"] - imbalance[(s,y)]["neg"] 
                                            for s in year_to_stations[y]),
                          reverse=True)

    # Loop per year
    for year in years_sorted:
        stations = year_to_stations[year]
        # Check overall dataset parity
        total_pos = sum(1 for rec in results if rec[2][4] >= 0.1)
        total_neg = sum(1 for rec in results if rec[2][4] < 0.1)
        if total_neg >= total_pos:
            logging.info("Overall dataset reached parity; stopping balancing.")
            break

        # For this year, track which stations still need negatives
        need = {}
        for station in stations:
            cnts = station_year_records.get((station, year), [])
            pos = sum(1 for v in cnts if v >= 0.1)
            neg = sum(1 for v in cnts if v < 0.1)
            if pos > neg:
                need[station] = {"pos": pos, "neg": neg}
        if not need:
            continue  # already balanced now

        logging.info(f"Year {year} needs balancing for stations: {need}")

        # Precompute avg locations & patch deg half-sizes
        locs = {}
        for station in need:
            avg_lat, avg_lon = station_year_loc[(station, year)]
            res_km = params["res_km"]
            pixel_count = params["pixel_count"]
            half_km = (pixel_count * res_km) / 2.0
            half_lat_deg = half_km / 111.0
            half_lon_deg = half_km / (111.0 * math.cos(math.radians(avg_lat)))
            locs[station] = (avg_lat, avg_lon, half_lat_deg, half_lon_deg)

        # Gather all granules in winter months for this year
        granule_items = []
        for (month_start, next_month) in winter_month_ranges(year):
            start_iso = month_start.strftime("%Y-%m-%dT00:00:00Z")
            end_iso = next_month.strftime("%Y-%m-%dT00:00:00Z")
            for sn in params["short_names"]:
                try:
                    res = earthaccess.search_data(
                        short_name=sn,
                        temporal=(start_iso, end_iso),
                        bounding_box=bbox
                    )
                    if res:
                        granule_items.extend(res)
                except Exception as e:
                    logging.warning(f"Search failed for {sn} in {year}-{month_start.month}: {e}")
        if not granule_items:
            logging.info(f"No winter granules found for year {year}. Cannot balance these stations.")
            continue

        # Extract unique granule filenames, filter out processed_set
        unique_items = []
        seen_fnames = set()
        for item in granule_items:
            fname = get_granule_filename(item)
            if not fname or fname in processed_set or fname in invalid_set or fname in seen_fnames:
                continue
            unique_items.append((item, fname))
            seen_fnames.add(fname)
        if not unique_items:
            logging.info(f"All winter granules already processed for year {year}. Skipping.")
            continue

        # Shuffle granules
        rand.shuffle(unique_items)

        attempts = 0
        # Iterate granules until all stations balanced or attempts exhausted
        for (item, fname) in unique_items:
            if not need:
                break
            if attempts >= max_granule_attempts:
                logging.info(f"Reached granule attempt cap ({max_granule_attempts}) for year {year}.")
                break
            attempts += 1

            # Download granule
            try:
                os.makedirs(temp_root, exist_ok=True)
                paths = earthaccess.download([item], temp_root)
                if not paths:
                    continue
                gran_path = paths[0]
            except Exception as e:
                logging.warning(f"Download failed for {fname}: {e}")
                continue

            # For PACE vs others:
            if sensor == "PACE":
                # Use process_pace_granule to get arr_stack etc.
                try:
                    result = process_pace_granule(gran_path, bbox, params, wave_all)
                except Exception as e:
                    logging.warning(f"PACE processing failed for {fname}: {e}")
                    with open(invalid_txt, "a") as f:
                        f.write(fname + "\n")
                    try: os.remove(gran_path)
                    except: pass
                    continue
                if result is None:
                    # no valid bands
                    os.remove(gran_path)
                    continue
                wls, arr_stack, target_lats, target_lons = result
                # Compute centers arrays:
                # target_lats and target_lons are 1D arrays of grid centers from regrid_pace_slice
                lat_centers = np.asarray(target_lats)
                lon_centers = np.asarray(target_lons)
            else:
                # Open & regrid full bbox once
                try:
                    with xr.open_dataset(gran_path, group="geophysical_data") as obs, \
                         xr.open_dataset(gran_path, group="navigation_data") as nav:
                        nav = nav.set_coords(("longitude", "latitude"))
                        dataset = xr.merge((obs, nav.coords))
                except Exception as e:
                    logging.warning(f"Failed to open granule {fname}: {e}")
                    os.remove(gran_path)
                    continue
                regridded = regrid_granule(dataset, bbox, params["res_km"])
                if regridded is None:
                    os.remove(gran_path)
                    continue

            # Determine granule datetime and year
            dt0 = extract_datetime_from_filename(fname)
            if dt0 is None:
                logging.info(f"Cannot parse datetime from {fname}; skipping.")
                os.remove(gran_path)
                continue
            gran_year = dt0.year
            if gran_year != year:
                os.remove(gran_path)
                continue

            # For each station still needing negatives, try patch
            to_remove = []
            for station, cnts in need.items():
                avg_lat, avg_lon, half_lat_deg, half_lon_deg = locs[station]
                if sensor == "PACE":
                    # Extract patch via extract_pace_patch
                    patch_dict = extract_pace_patch(
                        arr_stack, wls, avg_lon, avg_lat,
                        params["pixel_count"],
                        lat_centers=lat_centers,
                        lon_centers=lon_centers
                    )
                else:
                    patch = extract_patch_from_regridded(
                        regridded, avg_lon, avg_lat, params["pixel_count"], params["res_km"]
                    )
                    if patch is None:
                        patch_dict = None
                    else:
                        patch_dict, _, _ = patch

                if patch_dict is None:
                    continue  # this granule doesn't cover station well
                # Check if all-NaN:
                if all(np.isnan(arr).all() for arr in patch_dict.values()):
                    continue

                # Good patch: create negative sample
                t0_ts = pd.to_datetime(dt0).tz_localize("UTC")
                label_tuple = (station, t0_ts.to_pydatetime(), avg_lat, avg_lon, 0.0, np.nan)
                band_names = sorted(patch_dict.keys())
                patch_stack = np.stack([patch_dict[b] for b in band_names], axis=-1)
                patch_flat = patch_stack.flatten()
                results.append((fname, station, label_tuple, patch_flat))
                # Update counts
                station_year_records.setdefault((station, year), []).append(0.0)
                need_cnt = station_year_records[(station, year)]
                pos = sum(1 for v in need_cnt if v >= 0.1)
                neg = sum(1 for v in need_cnt if v < 0.1)
                logging.info(f"Added negative for {station} year {year} from granule {fname}; now pos={pos}, neg={neg}")
                if neg >= pos:
                    to_remove.append(station)
                # Also check overall parity
                total_pos = sum(1 for rec in results if rec[2][4] >= 0.1)
                total_neg = sum(1 for rec in results if rec[2][4] < 0.1)
                if total_neg >= total_pos:
                    logging.info("Overall dataset reached parity; stopping balancing entirely.")
                    to_remove = list(need.keys())
                    break
                # Continue: same granule might serve another station

            # Remove balanced stations
            for s in to_remove:
                need.pop(s, None)
            # Mark granule processed so not reused
            processed_set.add(fname)
            with open(processed_txt, "a") as f:
                f.write(fname + "\n")
            # Save updated results
            np.save(results_path, np.array(results, dtype=object))
            # Clean up granule file
            try:
                os.remove(gran_path)
            except:
                pass
            # If no stations left for this year, break
            if not need:
                logging.info(f"Station-year(s) for year {year} balanced; moving to next year.")
                break
        # End granule loop for year
        if need:
            logging.info(f"Year {year} ended with imbalance for stations: {need}.")
        # Clean up temp folder contents
        try:
            for fn in os.listdir(temp_root):
                os.remove(os.path.join(temp_root, fn))
        except:
            pass

    logging.info("Granule-first balancing complete.")
