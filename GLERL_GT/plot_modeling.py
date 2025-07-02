import os
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime, timedelta, timezone
import earthaccess
import logging
import xarray as xr
import numpy as np
import tensorflow as tf
from helpers import (
    regrid_granule,
    extract_datetime_from_filename,
    process_pace_granule,
    extract_pace_patch,
    plot_granule
)

start_date       = datetime(2024,4,15,tzinfo=timezone.utc)
end_date         = datetime(2024,9,1,tzinfo=timezone.utc)
bbox             = (-83.5, 41.3, -82.45, 42.2)
model_filename   = "./models/3x3_thres1.keras"
means_filename   = "channel_means.npy"
stds_filename    = "channel_stds.npy"
patch_size       = 3
wl_ref_file      = "./data/ref/PACE_OCI.20240603T180158.L2.OC_AOP.V3_0.nc"
out_dir          = "../Images/Daily_Plots"
dpi              = 100
window_size_days = 2
include_chla     = False

min_lon, min_lat, max_lon, max_lat = bbox

print("Loading CNN model and stats")
cnn = tf.keras.models.load_model(model_filename)
means = np.load(means_filename)
stds = np.load(stds_filename)

print("Loading wavelength list")
wave_all = xr.open_dataset(wl_ref_file, group="sensor_band_parameters")["wavelength_3d"].data

date = start_date.date()

while date <= end_date.date():

    print(f"building plot for {date}.")

    tspan = (date - timedelta(days = window_size_days), date)
    print(f"tspan: {tspan}.")

    if include_chla:
        # ===== CHL =====
        chl_results = earthaccess.search_data(
            short_name   = "OLCIS3A_L2_EFR_OC",
            temporal     = tspan,
            bounding_box = bbox
        )

        chl_fileset = earthaccess.download(chl_results, './data/')
        print(f"Got {len(chl_fileset)} files for this date.")

        chlarray = None
        valid_count = 0

        for i, file in enumerate(chl_fileset):
            
            print(f"regridding chl file {i+1} of {len(chl_fileset)}.")

            # load navigation and chlorophyll
            nav_chl = (
                xr.open_dataset(file, group="navigation_data")
                .set_coords(("longitude", "latitude"))
            )
            if "pixel_control_points" in nav_chl.dims:
                nav_chl = nav_chl.rename_dims({"pixel_control_points": "pixels_per_line"})

            ds_chl = xr.open_dataset(file, group="geophysical_data")

            chlor_a = xr.merge((ds_chl["chlor_a"], nav_chl))
            
            chlor_a = chlor_a.where((
                (chlor_a["latitude"] > bbox[1]) & \
                (chlor_a["latitude"] < bbox[3]) & \
                (chlor_a["longitude"] > bbox[0]) & \
                (chlor_a["longitude"] < bbox[2])),
            drop = True)

            regrid = regrid_granule(chlor_a, bbox, 0.3, chlor_a=True)

            if regrid is not None:
                block = regrid["chlor_a"]
                valid_count += 1
            else:
                continue
            
            if not isinstance(chlarray, xr.DataArray):
                if chlarray is None:
                    # uninitialized
                    chlarray = block
                else:
                    raise ValueError(f"Expected a DataArray or None, got {type(chlarray)}")
            else:
                # safe to do array math
                block = block.fillna(0)
                chlarray = chlarray + block

        chlarray = chlarray / valid_count
        chlarray = chlarray.where(chlarray != 0, other=np.nan)

    # ===== MC =====

    mc_results = earthaccess.search_data(
        short_name  = "PACE_OCI_L2_AOP",
        temporal    = tspan,
        bounding_box= bbox
    )    

    mc_results_NRT = earthaccess.search_data(
        short_name  = "PACE_OCI_L2_AOP_NRT",
        temporal    = tspan,
        bounding_box= bbox
    )

    mc_results = mc_results + mc_results_NRT

    if len(mc_results) == 0:
        print("No granules found in search.")
        date += timedelta(days=1)
        continue

    mc_fileset = earthaccess.download(mc_results, './data/')

    pacearray = None
    wls      = None

    valid_count = 0
    for i, file in enumerate(mc_fileset):
        print(f"regridding pace file {i+1} of {len(mc_fileset)} - {file}")

        try:
            res = process_pace_granule(file, bbox, {"res_km": 1.2}, wave_all)
        except Exception as e:
            logging.warning(f"Skipping granule {file!r} due to error: {e}")
            continue

        if not res:
            logging.warning("no valid data → skipping")
        elif ((wls is not None) and (lat_c is not None) and (lon_c is not None) and \
             (not res[0] == wls or \
              not all(res[2] == lat_c) or \
              not all(res[3] == lon_c))):
            logging.warning("results don't match")
            exit()
        else:
            valid_count += 1
            wls, arr_stack, lat_c, lon_c = res
            plot_granule(file, arr_stack, bbox, out_dir)
            if not isinstance(pacearray, np.ndarray):
                if pacearray is None:
                    print("initializing pacearray")
                    pacearray = np.nan_to_num(arr_stack, nan=0.0)
                    valid_pixel_count = (~np.isnan(arr_stack)).astype(int)
            else:
                # sum where valid
                arr_stack_filled = np.nan_to_num(arr_stack, nan=0.0)
                temp = pacearray
                pacearray = np.add(temp, arr_stack_filled)
                print("pacearray == temp?", np.all((pacearray == temp) | (np.isnan(pacearray) & np.isnan(temp))))
                
                # increment count where valid
                valid_pixel_count += (~np.isnan(arr_stack)).astype(int)

    if valid_count == 0:
        print("No valid microcystin granules found for this date")
        date += timedelta(days=1)
        continue

    # avoid divide by zero
    with np.errstate(invalid="ignore", divide="ignore"):
        pacearray = pacearray / valid_pixel_count

    # replace 0 counts with nan
    pacearray[valid_pixel_count == 0] = np.nan

    patches = []
    coords = []

    r_idx, g_idx, b_idx = 105, 75, 48

    # extract each band and transpose so that we get (H, W)
    r = pacearray[r_idx, :, :].T
    g = pacearray[g_idx, :, :].T
    b = pacearray[b_idx, :, :].T

    # stack into (H, W, 3)
    rgb = np.dstack((r, g, b))

    # normalize to [0,1] for display
    rgb_min, rgb_max = np.nanmin(rgb), np.nanmax(rgb)
    rgb_norm = (rgb - rgb_min) / (rgb_max - rgb_min)

    for i, la in enumerate(lat_c):
        for j, lo in enumerate(lon_c):
            pd = extract_pace_patch(pacearray, wls, lo, la, patch_size, lat_c, lon_c)
            bs = np.stack([pd[wl] for wl in wls], axis=-1)
            
            mask = np.any(~np.isnan(bs), axis=-1, keepdims=True).astype("float32")
            
            # Skip fully zero/NaN patches
            if mask.sum() == 0:
                continue

            p4 = np.concatenate([bs, mask], axis=-1)
            channel0 = p4[...,0].ravel()
            p4 = (p4 - means.reshape(1,1,-1)) / (stds.reshape(1,1,-1) + 1e-6)
            p4 = np.nan_to_num(p4, nan=0.0)
            
            patches.append(p4)
            coords.append((i,j))
    
    if not patches:
        raise RuntimeError("No valid patches found after averaging")
    
    X = np.stack(patches, axis=0)

    probs = cnn.predict(X).ravel()
    raw_preds = cnn.predict(X)
    
    #probs[probs < 0.05] = np.nan # or use np.nan
    mc_map = np.full((len(lat_c), len(lon_c)), np.nan)
    count = 0
    for (i, j), p in zip(coords, probs):
        mc_map[i, j] = p

    fig, ax = plt.subplots(figsize=(8,6),
                       subplot_kw={"projection": ccrs.PlateCarree()})

    # Basemap
    ax.set_facecolor("black")
    ax.add_feature(cfeature.LAND.with_scale("10m"),
                   facecolor="white", edgecolor="none")
    ax.add_feature(cfeature.LAKES.with_scale("10m"),
                   facecolor="black", zorder = 0)
    
    # Chlor plot
    if include_chla:
        chlor_a_plot = np.log10(chlarray).plot.pcolormesh(
            ax=ax,
            cmap="summer",
            vmin=0.5, vmax=1.5,
        transform=ccrs.PlateCarree(),
            zorder=5,
            add_colorbar=True,
            cbar_kwargs={
                "shrink": 0.8,
                "pad": 0.05,
                "label": "Chlorophyll-a [log10 (mg m$^{-3}$)]"
            }
        )

    extent = [bbox[0], bbox[2], bbox[1], bbox[3]]
    rgb_rot = np.transpose(rgb_norm, (1, 0, 2))

    ax.imshow(
        rgb_rot,
        origin="lower",
        extent=extent,
        transform=ccrs.PlateCarree(),
        zorder=2,         # behind pcolormesh
        interpolation="nearest"
    )
    
    # MC plot.
    ax.imshow(
        mc_map,
        origin="lower",
        extent=extent,                    # same [lon_min, lon_max, lat_min, lat_max]
        transform=ccrs.PlateCarree(),
        cmap="Reds",
        vmin=0, vmax=1,
        alpha = np.where((np.isnan(mc_map) | (mc_map < 0.5)), 0.0, 0.667),
        interpolation="nearest",
        zorder=10
    )

    # add coastlines, gridlines, colorbar as before…
    ax.coastlines(resolution="10m")
    gl = ax.gridlines(draw_labels=True, linestyle="--", linewidth=0.5)
    gl.top_labels = False; gl.right_labels = False

    # if you still want a colorbar for MC:
    mappable = plt.cm.ScalarMappable(cmap="Reds", norm=plt.Normalize(0,1))
    mappable.set_array(mc_map)  # so the colorbar knows the data range
    cbar = plt.colorbar(mappable, ax=ax, orientation="vertical", pad=0.02, shrink=0.7)
    cbar.set_label("Microcystin Probability")

    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{date.strftime('%Y%m%d')}.png"),
                dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    date += timedelta(days=1)

    for filename in os.listdir('./data/'):
        if filename.endswith('.nc'):
            file_path = os.path.join('./data/', filename)
            try:
                os.remove(file_path)
                print(f"Removed: {file_path}")
            except OSError as e:
                print(f"Error removing {file_path}: {e}")

print("done!")