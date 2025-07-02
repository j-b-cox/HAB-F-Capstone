import os
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import xarray as xr
import earthaccess
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

from helpers import *
from balance_training_data import balance_dataset_by_granule

def process_granule(filepath, station_df, bbox, sensor_params,
                    processed_granules_set, results, station_colors, color_cycle,
                    results_path, processed_txt_path, images_root, wave_all):
    filename = os.path.basename(filepath)
    if filename in processed_granules_set:
        logging.info(f"Granule {filename} already processed; skipping.")
        try: os.remove(filepath)
        except: pass
        return

    dt = extract_datetime_from_filename(filename)
    if dt is None:
        logging.warning(f"Cannot extract datetime from {filename}; marking processed.")
        processed_granules_set.add(filename)
        with open(processed_txt_path, "a") as f:
            f.write(filename + "\n")
        try: os.remove(filepath)
        except: pass
        return

    if sensor_params.get("sensor") == "PACE":
        try:
            result = process_pace_granule(filepath, bbox, sensor_params, wave_all)
            if result is None:
                logging.warning(f"No valid PACE data in {filename}; skipping.")
                # mark processed, clean up, and return
                return
            wls, arr_stack, target_lats, target_lons = result

        except Exception as e:
            logging.error(f"PACE processing failed for {filename}: {e}")
            processed_granules_set.add(filename)
            #with open(processed_txt_path, "a") as f:
            #    f.write(filename + "\n")
            #try: os.remove(filepath)
            #except: pass
            return
        if arr_stack is None:
            logging.warning(f"No valid PACE bands for {filename}")
            processed_granules_set.add(filename)
            with open(processed_txt_path, "a") as f:
                f.write(filename + "\n")
            try: os.remove(filepath)
            except: pass
            return
        # Now arr_stack is shape (n_wl, ny, nx). Next: for each station within ±2d, extract patch:

        # Subset station rows ±2 days
        t0 = pd.to_datetime(dt).tz_localize('UTC')
        window_start = t0 - timedelta(days=2)
        window_end   = t0 + timedelta(days=2)
        mask_time = (station_df["timestamp"] >= window_start) & (station_df["timestamp"] <= window_end)
        df_subset = station_df.loc[mask_time]
        if df_subset.empty:
            logging.info(f"No station observations within ±2 days of {filename} at {t0}; marking processed.")
            processed_granules_set.add(filename)
            with open(processed_txt_path, "a") as f:
                f.write(filename + "\n")
            try: os.remove(filepath)
            except: pass
            return

        # For each station:
        station_patches_for_plot = []
        for station in df_subset["station_name"].unique():
            
            df_st = df_subset[df_subset["station_name"] == station]
            times = df_st["timestamp"].tolist()

            lats = df_st["lat"].values
            lons = df_st["lon"].values
            
            res_deg = sensor_params["res_km"] / 111.0 

            res_lat_deg = sensor_params["res_km"] / 111.0
            # if you need lon resolution separately:
            lat0 = (bbox[1] + bbox[3]) / 2.0
            
            res_lon_deg = sensor_params["res_km"] / (111.0 * math.cos(math.radians(lat0)))
            # Then use res_lat_deg, res_lon_deg when building the grid.

            lon_min, lat_min, lon_max, lat_max = bbox
            lat_bins = np.arange(lat_min, lat_max + res_lat_deg, res_lat_deg)
            lon_bins = np.arange(lon_min, lon_max + res_lon_deg, res_lon_deg)

            lat_centers = 0.5 * (lat_bins[:-1] + lat_bins[1:])
            lon_centers = 0.5 * (lon_bins[:-1] + lon_bins[1:])

            lat0, lon0 = estimate_position(times, lats, lons, t0)
            patch_dict = extract_pace_patch(arr_stack, wls, lon0, lat0,
                                            pixel_count = 1,
                                            lat_centers=lat_centers,
                                            lon_centers=lon_centers)  # or convert km to deg approx
            
            if all(np.isnan(arr).all() for arr in patch_dict.values()):
                print(f"No data for station {station}")
                continue

            mean_micro = float(df_st["particulate_microcystin"].dropna().mean()) if "particulate_microcystin" in df_st.columns else np.nan
            mean_diss  = float(df_st["dissolved_microcystin"].dropna().mean()) if "dissolved_microcystin" in df_st.columns else np.nan
            mean_chla  = float(df_st["extracted_chla"].dropna().mean()) if "extracted_chla" in df_st.columns else np.nan            # Build patch_stack: stack in wavelength-sorted order:
            band_wls = sorted(patch_dict.keys())
            patch_stack = np.stack([patch_dict[wl] for wl in band_wls], axis=-1)  # shape (px, px, n_wl)
            patch_flat = patch_stack.flatten()
            results.append((filename, station, (station, t0, lat0, lon0, mean_micro, mean_diss, mean_chla), patch_flat))
            logging.info(f"Saved patch for granule {filename}, station {station}")
            
            # For plotting, store patch_dict etc.
            color = assign_color_for_station(station, station_colors, color_cycle)
            station_patches_for_plot.append({
                "station_name": station,
                "lon0": lon0,
                "lat0": lat0,
                "half_lon_deg": (sensor_params["pixel_count"]//2) * sensor_params["res_km"] / 111.0,  # approximate deg
                "half_lat_deg": (sensor_params["pixel_count"]//2) * sensor_params["res_km"] / 111.0,
                "color": color,
                "mean_micro": mean_micro,
                "patch_dict": patch_dict
            })

        # Mark processed, save results, then plotting:
        processed_granules_set.add(filename)
        with open(processed_txt_path, "a") as f:
            f.write(filename + "\n")
        save_station_colors(sensor_params["station_colors_json"], station_colors)
        np.save(results_path, np.array(results, dtype=object))

        # Plot overview image
        if station_patches_for_plot:
            try:
                # Create figure with 2 rows: map (larger), spectral (smaller)
                
                # Choose target wavelengths for true-color:
                rgb_wls = {"R": 645.0, "G": 555.0, "B": 450.0}
                data_vars = {}
                for _, target_wl in rgb_wls.items():
                    # find nearest index
                    wls = np.asarray(wls)  # if wls was a list or tuple of numbers
                    idx = int(np.argmin(np.abs(wls - target_wl)))
                    actual_wl = wls[idx]
                    arr2d = arr_stack[idx]  # shape (ny, nx)
                    varname = f"Rrs_{int(round(actual_wl))}"

                    da = xr.DataArray(
                        arr2d,
                        dims=("latitude", "longitude"),
                        coords={"latitude": target_lats, "longitude": target_lons},
                        name=varname
                    )
                    data_vars[varname] = da

                if data_vars:
                    regridded = xr.Dataset(data_vars)
                else:
                    regridded = None

                fig = plt.figure(figsize=(10, 12), dpi=150)
                # Top: map
                ax_map = fig.add_subplot(2, 1, 1, projection=ccrs.PlateCarree())
                ax_map.set_extent([bbox[0], bbox[2], bbox[1], bbox[3]], crs=ccrs.PlateCarree())
                # Plot true-color composite
                plot_true_color(ax_map, regridded, bbox, station_patches_for_plot, station_colors)
                # Overlay station patches
                for sp in station_patches_for_plot:
                    rect = plt.Rectangle(
                        (sp["lon0"] - sp["half_lon_deg"], sp["lat0"] - sp["half_lat_deg"]),
                        2*sp["half_lon_deg"], 2*sp["half_lat_deg"],
                        edgecolor=sp["color"], facecolor="none", transform=ccrs.PlateCarree(),
                        linewidth=1.5
                    )
                    ax_map.add_patch(rect)
                    ax_map.text(
                        sp["lon0"], sp["lat0"], sp["station_name"],
                        color=sp["color"], transform=ccrs.PlateCarree(),
                        fontsize=6, ha="center", va="center"
                    )
                # Add features
                ax_map.coastlines(resolution='10m', linewidth=0.5)
                ax_map.add_feature(cfeature.BORDERS, linewidth=0.5)
                ax_map.add_feature(cfeature.STATES, linewidth=0.5)
                ax_map.xaxis.set_major_formatter(LongitudeFormatter())
                ax_map.yaxis.set_major_formatter(LatitudeFormatter())
                ax_map.set_title(f"Granule {filename} @ {t0.isoformat()}")

                # Bottom: one spectral curve per station patch
                ax_spec = fig.add_subplot(2, 1, 2)
                for sp in station_patches_for_plot:
                    station = sp["station_name"]
                    color = sp["color"]
                    mean_micro = sp.get("mean_micro", 0.0)
                    patch_dict = sp.get("patch_dict", {})
                    if not patch_dict:
                        continue

                    # 1. Sort numeric wavelengths
                    wavelengths = sorted(patch_dict.keys())  # e.g. [443.0, 490.0, 555.0, ...]
                    means = []
                    for wl in wavelengths:
                        arr = patch_dict[wl]
                        flat = arr.flatten()
                        valid = flat[~np.isnan(flat)]
                        if valid.size:
                            means.append(np.nanmean(valid))
                        else:
                            means.append(np.nan)  # or skip
                    wls = np.array(wavelengths)
                    ms = np.array(means)

                    # 2. Choose marker based on mean_micro threshold
                    if mean_micro >= 0.1:
                        marker = '^'
                        markersize = 8
                    else:
                        marker = 'o'
                        markersize = 5

                    # 3. Plot
                    ax_spec.plot(
                        wls, ms,
                        marker=marker,
                        markersize=markersize,
                        linestyle='-',
                        label=f"{station} ({mean_micro:.2f})",
                        color=color
                    )

                # After looping:
                if station_patches_for_plot:
                    ax_spec.set_xlabel("Wavelength (nm)")
                    ax_spec.set_ylabel("Mean Rrs (patch)")
                    ax_spec.set_title("Spectral curves per station patch")
                    ax_spec.grid(True, linewidth=0.3, alpha=0.5)
                    ax_spec.legend(fontsize='small', loc='best')
                else:
                    ax_spec.text(0.5, 0.5, "No station patches", ha='center', va='center')
                    ax_spec.set_axis_off()

                # Save figure
                images_dir = os.path.join(images_root, t0.strftime("%Y_%m"))
                os.makedirs(images_dir, exist_ok=True)
                outpath = os.path.join(images_dir, f"{os.path.splitext(filename)[0]}.png")
                #plt.tight_layout()
                plt.savefig(outpath)#, bbox_inches="tight")
                plt.close(fig)
            except Exception as e:
                logging.error(f"Error plotting overview for {filename}: {e}")
        else:
            logging.info(f"No station patches for granule {filename}; skipping plot.")
        
        # Delete granule file
        try:
            os.remove(filepath)
        except Exception as e:
            logging.warning(f"Could not delete granule file {filepath}: {e}")
        return
    else:
        # Open dataset
        try:
            with xr.open_dataset(filepath, group="geophysical_data") as obs, \
                xr.open_dataset(filepath, group="navigation_data") as nav:
                nav = nav.set_coords(("longitude", "latitude"))
                dataset = xr.merge((obs, nav.coords))
        except Exception as e:
            logging.error(f"Failed to open granule {filename}: {e}; marking processed.")
            processed_granules_set.add(filename)
            with open(processed_txt_path, "a") as f:
                f.write(filename + "\n")
            try: os.remove(filepath)
            except: pass
            return
        regridded = regrid_granule(dataset, bbox, sensor_params["res_km"])
        if regridded is None:
            logging.warning(f"Regridding failed or no bands for granule {filename}; marking processed.")
            processed_granules_set.add(filename)
            with open(processed_txt_path, "a") as f:
                f.write(filename + "\n")
            try: os.remove(filepath)
            except: pass
            return

        # Subset station rows ±2 days
        t0 = pd.to_datetime(dt).tz_localize('UTC')
        window_start = t0 - timedelta(days=2)
        window_end   = t0 + timedelta(days=2)
        mask_time = (station_df["timestamp"] >= window_start) & (station_df["timestamp"] <= window_end)
        df_subset = station_df.loc[mask_time]
        if df_subset.empty:
            logging.info(f"No station observations within ±2 days of {filename} at {t0}; marking processed.")
            processed_granules_set.add(filename)
            with open(processed_txt_path, "a") as f:
                f.write(filename + "\n")
            try: os.remove(filepath)
            except: pass
            return

        print("Resetting station_patches_for_plot")
        station_patches_for_plot = []
        for station in df_subset["station_name"].unique():
            df_st = df_subset[df_subset["station_name"] == station]
            times = df_st["timestamp"].tolist()
            lats = df_st["lat"].values
            lons = df_st["lon"].values
            lat0, lon0 = estimate_position(times, lats, lons, t0)
            patch_dict, half_lon_deg, half_lat_deg = extract_patch_from_regridded(
                regridded, lon0, lat0, sensor_params["pixel_count"], sensor_params["res_km"]
            )
            if patch_dict is None:
                continue
            mean_micro = float(df_st["particulate_microcystin"].dropna().mean()) if "particulate_microcystin" in df_st.columns else np.nan
            mean_diss  = float(df_st["dissolved_microcystin"].dropna().mean()) if "dissolved_microcystin" in df_st.columns else np.nan
            mean_chla  = float(df_st["extracted_chla"].dropna().mean()) if "extracted_chla" in df_st.columns else np.nan
            label_tuple = (station, t0, lat0, lon0, mean_micro, mean_diss, mean_chla)
            band_names = sorted(patch_dict.keys())
            patch_stack = np.stack([patch_dict[b] for b in band_names], axis=-1)
            patch_flat = patch_stack.flatten()
            results.append((filename, station, label_tuple, patch_flat))
            logging.info(f"Saved patch for granule {filename}, station {station}")
            color = assign_color_for_station(station, station_colors, color_cycle)
            station_patches_for_plot.append({
                "station_name": station,
                "lon0": lon0,
                "lat0": lat0,
                "half_lon_deg": half_lon_deg,
                "half_lat_deg": half_lat_deg,
                "color": color,
                "mean_micro": mean_micro,
                "patch_dict": patch_dict
            })

        # Mark processed
        processed_granules_set.add(filename)
        with open(processed_txt_path, "a") as f:
            f.write(filename + "\n")
        save_station_colors(sensor_params["station_colors_json"], station_colors)
        np.save(results_path, np.array(results, dtype=object))

        # Plot overview image
        if station_patches_for_plot:
            try:
                # Create figure with 2 rows: map (larger), spectral (smaller)
                # Top subplot: map with PlateCarree projection
                # Bottom subplot: spectral (no projection)
                fig = plt.figure(figsize=(15, 18), dpi=150)
                # Top: map
                ax_map = fig.add_subplot(2, 1, 1, projection=ccrs.PlateCarree())
                ax_map.set_extent([bbox[0], bbox[2], bbox[1], bbox[3]], crs=ccrs.PlateCarree())
                # Plot true-color composite
                plot_true_color(ax_map, regridded, bbox, station_patches_for_plot, station_colors)
                # Overlay station patches
                for sp in station_patches_for_plot:
                    rect = plt.Rectangle(
                        (sp["lon0"] - sp["half_lon_deg"], sp["lat0"] - sp["half_lat_deg"]),
                        2*sp["half_lon_deg"], 2*sp["half_lat_deg"],
                        edgecolor=sp["color"], facecolor="none", transform=ccrs.PlateCarree(),
                        linewidth=1.5
                    )
                    ax_map.add_patch(rect)
                    ax_map.text(
                        sp["lon0"], sp["lat0"], sp["station_name"],
                        color=sp["color"], transform=ccrs.PlateCarree(),
                        fontsize=6, ha="center", va="center"
                    )
                # Add features
                ax_map.coastlines(resolution='10m', linewidth=0.5)
                ax_map.add_feature(cfeature.BORDERS, linewidth=0.5)
                ax_map.add_feature(cfeature.STATES, linewidth=0.5)
                ax_map.xaxis.set_major_formatter(LongitudeFormatter())
                ax_map.yaxis.set_major_formatter(LatitudeFormatter())
                ax_map.set_title(f"Granule {filename} @ {t0.isoformat()}")

                # Bottom: one spectral curve per station patch
                ax_spec = fig.add_subplot(2, 1, 2)
                for sp in station_patches_for_plot:
                    station = sp["station_name"]
                    color = sp["color"]
                    mean_micro = sp.get("mean_micro", 0.0)
                    patch_dict = sp.get("patch_dict", {})
                    wavelengths = []
                    means = []
                    for band_name, arr in patch_dict.items():
                        parts = band_name.split("_")
                        try:
                            wl = float(parts[-1])
                        except:
                            continue
                        flat = arr.flatten()
                        valid = flat[~np.isnan(flat)]
                        if valid.size == 0:
                            continue
                        mean_val = np.nanmean(valid)
                        wavelengths.append(wl)
                        means.append(mean_val)
                    if not wavelengths:
                        continue
                    idx = np.argsort(wavelengths)
                    wls = np.array(wavelengths)[idx]
                    ms = np.array(means)[idx]
                    # Choose marker: triangle if mean_micro >= 0.1
                    if mean_micro >= 0.1:
                        marker = '^'
                        markersize = 9  # 50% larger than default circles; adjust if needed
                    else:
                        marker = 'o'
                        markersize = 6
                    ax_spec.plot(wls, ms, marker=marker, markersize=markersize,
                                linestyle='-', label=station, color=color)
                # Finalize
                if station_patches_for_plot:
                    ax_spec.set_xlabel("Wavelength (nm)")
                    ax_spec.set_ylabel("Mean Rrs (patch)")
                    ax_spec.set_title("Spectral curves per station patch")
                    ax_spec.grid(True, linewidth=0.3, alpha=0.5)
                    ax_spec.legend(fontsize='small', loc='best')
                else:
                    ax_spec.text(0.5, 0.5, "No station patches", ha='center', va='center')
                    ax_spec.set_axis_off()

                # Save figure
                images_dir = os.path.join(images_root, t0.strftime("%Y_%m"))
                os.makedirs(images_dir, exist_ok=True)
                outpath = os.path.join(images_dir, f"{os.path.splitext(filename)[0]}.png")
                plt.tight_layout()
                plt.savefig(outpath, bbox_inches="tight")
                plt.close(fig)
            except Exception as e:
                logging.error(f"Error plotting overview for {filename}: {e}")
        else:
            logging.info(f"No station patches for granule {filename}; skipping plot.")
        
        # Delete granule file
        try:
            os.remove(filepath)
        except Exception as e:
            logging.warning(f"Could not delete granule file {filepath}: {e}")

    

def process_all_granules(sensor="SENTINEL", test_year=None, test_month=None):
    configure_logging()
    logging.info(f"Starting granule-first processing for sensor={sensor}")
    df = pd.read_csv("glrl-hab-data.csv", index_col=0)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    latest_ts = df["timestamp"].max()
    if pd.isna(latest_ts):
        logging.warning("No valid timestamps in station DataFrame; nothing to process.")
        return
    # Get date from Timestamp (UTC-aware); Timestamp.date() returns the date portion
    end_date = latest_ts.date()

    sensor = sensor.upper()
    sensor_map = {
        "MODIS": {
            "short_names": ["MODISA_L2_OC", "MODIST_L2_OC"],
            "start_date": None,
            "res_km": 1.0,
            "pixel_count": 3,
        },
        "SENTINEL": {
            "short_names": [
                "OLCIS3A_L2_EFR_OC", "OLCIS3A_L2_EFR_OC_NRT",
                "OLCIS3B_L2_EFR_OC", "OLCIS3B_L2_EFR_OC_NRT"],
            "start_date": datetime(2018,5,1),
            "res_km": 0.4,
            "pixel_count": 8,
        },
        "PACE": {
            "short_names": ["PACE_OCI_L2_AOP", "PACE_OCI_L2_AOP_NRT"],
            "start_date": datetime(2024,2,1),
            "res_km": 1.2,
            "pixel_count": 1,
        }
    }
    if sensor not in sensor_map:
        raise ValueError(f"Unsupported sensor '{sensor}'")
    params = sensor_map[sensor]
    params["sensor"] = sensor
    if sensor == "MODIS":
        earliest = df["timestamp"].min()
        params["start_date"] = earliest.to_pydatetime() if not pd.isna(earliest) else datetime.utcnow()
    params["bbox"] = (-83.5, 41.3, -82.45, 42.2)
    root_data = "./data"
    sensor_data_root = os.path.join(root_data, sensor)
    os.makedirs(sensor_data_root, exist_ok=True)
    processed_txt = f"processed_granules_{sensor}.txt"
    results_path = f"training_data_{sensor}.npy"
    station_colors_json = f"station_colors_{sensor}.json"
    images_root = os.path.join("Images", sensor)
    os.makedirs(images_root, exist_ok=True)
    params["station_colors_json"] = station_colors_json
    from matplotlib import cm
    tab20 = [cm.tab20(i) for i in range(20)]
    def rgba_to_hex(rgba):
        return '#{:02x}{:02x}{:02x}'.format(int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255))
    params["color_cycle"] = [rgba_to_hex(c) for c in tab20]
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


    if os.path.exists(results_path):
        existing = np.load(results_path, allow_pickle=True)
        results = existing.tolist()
    else:
        results = []
    station_colors = load_station_colors(station_colors_json)
    auth = earthaccess.login(persist=True)
    
    start = params["start_date"]
    if isinstance(start, datetime):
        start_date = start.date()
    else:
        start_date = start

    if start_date > end_date:
        logging.info(f"Start date {start_date} is after latest data date {end_date}; nothing to process.")
        return

    # For test month validation and month_starts, use end_date instead of today
    if test_year is not None and test_month is not None:
        month_start = datetime(test_year, test_month, 1)
        if not (start_date <= month_start.date() <= end_date):
            raise ValueError("Test month outside valid timeframe")
        month_starts = [month_start]
    else:
        # Generate month starts from start_date to end_date
        month_starts = pd.date_range(start=start_date, end=end_date, freq="MS").to_pydatetime()

    if sensor == "PACE":
        print("Retrieving wavelength list from a reference file...")
        search_ref = earthaccess.search_data(
            short_name="PACE_OCI_L2_AOP",
            temporal=("2024-06-01", "2024-06-05"),
            bounding_box=params["bbox"],
        )
        if not search_ref:
            raise RuntimeError("No reference files found to retrieve wavelengths.")
        ref_file = earthaccess.download(search_ref, "./Data/")[0]
        wave_all = xr.open_dataset(ref_file, group="sensor_band_parameters")["wavelength_3d"].data
    else:
        wave_all = None

    for ms in month_starts:
        year = ms.year; month = ms.month
        month_start = datetime(year, month, 1)
        # Compute next_month as before
        if month == 12:
            next_month = datetime(year+1, 1, 1)
        else:
            next_month = datetime(year, month+1, 1)
        # Cap next_month so it does not exceed end_date + 1 day
        if next_month.date() > end_date:
            next_month = datetime(end_date.year, end_date.month, end_date.day) + timedelta(days=1)
        start_iso = month_start.strftime("%Y-%m-%dT%H:%M:%SZ")
        end_iso   = next_month.strftime("%Y-%m-%dT%H:%M:%SZ")
        logging.info(f"Processing month {year}-{month:02d} for sensor {sensor}")

        all_search = []
        start_iso = month_start.strftime("%Y-%m-%dT%H:%M:%SZ")
        end_iso   = next_month.strftime("%Y-%m-%dT%H:%M:%SZ")
        for sn in params["short_names"]:
            try:
                res = earthaccess.search_data(
                    short_name=sn,
                    temporal=(start_iso, end_iso),
                    bounding_box=params["bbox"]
                )
                if res:
                    logging.info(f"Found {len(res)} granules for short_name={sn} in {year}-{month:02d}")
                    all_search.extend(res)
                else:
                    logging.info(f"No granules for short_name={sn} in {year}-{month:02d}")
            except Exception as e:
                logging.error(f"Search failed for {sn} in {year}-{month:02d}: {e}")
        if not all_search:
            logging.info(f"No granules found for month {year}-{month:02d}.")
            continue
        
        month_folder = os.path.join(sensor_data_root, f"{year}_{month:02d}")
        os.makedirs(month_folder, exist_ok=True)
        
        filtered_search = []
        skipped_processed = 0
        skipped_no_obs = 0

        for item in all_search:
            fname = get_granule_filename(item)
            if fname is None:
                logging.warning("Cannot determine filename for item; skipping")
                continue
            # If this granule was already processed, skip
            if fname in processed_set or fname in invalid_set:
                skipped_processed += 1
                continue

            # 2. Determine granule datetime for station-window filtering
            dt = None
            if hasattr(item, "time_start"):
                try:
                    dt = pd.to_datetime(item.time_start, utc=True)
                except:
                    dt = None
            if dt is None:
                # fallback: parse from fname
                dt0 = extract_datetime_from_filename(fname)
                if dt0 is not None:
                    dt = pd.to_datetime(dt0).tz_localize("UTC")
            if dt is None:
                logging.info(f"Skipping granule {fname}: cannot get datetime")
                continue

            # 3. Check for any station timestamps within ±2 days
            window_start = dt - pd.Timedelta(days=2)
            window_end   = dt + pd.Timedelta(days=2)
            if not ((df["timestamp"] >= window_start) & (df["timestamp"] <= window_end)).any():
                skipped_no_obs += 1
                continue

            # If passed both checks, keep this item
            filtered_search.append(item)

        logging.info(f"Skipped {skipped_processed} already-processed granules; "
                    f"{skipped_no_obs} with no nearby observations; "
                    f"{len(filtered_search)} remain for download.")

        if not filtered_search:
            logging.info("No relevant new granules to download for this month.")
            continue

        # Now download only filtered_search
        try:
            paths = earthaccess.download(filtered_search, month_folder)
            logging.info(f"Downloaded {len(paths)} granules into {month_folder}")
        except Exception as e:
            logging.error(f"Download failed: {e}")
            continue

        for filepath in paths:
            process_granule(
                filepath, df, params["bbox"], params,
                processed_set, results, station_colors, params["color_cycle"],
                results_path, processed_txt, images_root, wave_all
            )
        try:
            os.rmdir(month_folder)
            logging.info(f"Deleted month folder {month_folder}")
        except OSError:
            for fname in os.listdir(month_folder):
                fpath = os.path.join(month_folder, fname)
                try: os.remove(fpath)
                except: pass
            try: os.rmdir(month_folder)
            except: pass
            logging.info(f"Cleaned up month folder {month_folder}")
    logging.info(f"Finished processing all months for sensor={sensor}. Total patches: {len(results)}")

if __name__ == "__main__":
    # Example: change sensor here
    logging.basicConfig(level=logging.INFO)
    process_all_granules(sensor="PACE")#, test_year = 2024, test_month = 7)
    balance_dataset_by_granule("PACE")