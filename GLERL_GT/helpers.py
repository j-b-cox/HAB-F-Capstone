import os
import re
import json
import math
import logging

import numpy  as np
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from datetime import datetime
from pyresample import geometry, kd_tree
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

def extract_pace_patch(arr_stack, wavelengths, lon0, lat0, pixel_count, res_deg, lat_centers, lon_centers):
    """
    Given arr_stack shape (n_wl, ny, nx) on a regular lat/lon grid with resolution res_deg,
    extract a square patch of size pixel_count x pixel_count around (lat0, lon0) for each wavelength.
    Returns patch_dict mapping wavelength -> 2D array of shape (patch_count, patch_count).
    """
    import numpy as np
    # Here we assume global arrays lat_centers, lon_centers are accessible (or pass them in).
    # Find nearest index:
    lat_idx = np.abs(lat_centers - lat0).argmin()
    lon_idx = np.abs(lon_centers - lon0).argmin()
    half = pixel_count // 2
    patch_dict = {}
    ny, nx = arr_stack.shape[1], arr_stack.shape[2]
    for i, wl in enumerate(wavelengths):
        # Define slice indices, with bounds checking
        i0 = max(0, lat_idx - half)
        i1 = min(ny, lat_idx + half + 1)
        j0 = max(0, lon_idx - half)
        j1 = min(nx, lon_idx + half + 1)
        patch = arr_stack[i, i0:i1, j0:j1]

        # If patch not full size, you may pad with NaNs to get exactly pixel_count x pixel_count:
        if patch.shape != (pixel_count, pixel_count):
            pad_y = pixel_count - patch.shape[0]
            pad_x = pixel_count - patch.shape[1]
            pad_before_y = max(0, (pixel_count - patch.shape[0])//2)
            pad_before_x = max(0, (pixel_count - patch.shape[1])//2)
            pad_after_y = pad_y - pad_before_y
            pad_after_x = pad_x - pad_before_x
            patch = np.pad(patch,
                           ((pad_before_y, pad_after_y), (pad_before_x, pad_after_x)),
                           constant_values=np.nan)
        patch_dict[float(wl)] = patch  # key = numeric wavelength
    #if all(np.isnan(arr).all() for arr in patch_dict.values()): return None
    #else: return patch_dict
    return patch_dict

def process_pace_granule(filepath, bbox, sensor_params, wave_all):
    """
    Open a PACE granule, regrid its Rrs 3D variable onto the target bbox/resolution,
    and return an xarray DataArray of shape (wavelength, y, x) with reflectance.
    """

    # 1. Open dataset groups
    try:
        with xr.open_dataset(filepath, group="geophysical_data") as geo_ds, \
             xr.open_dataset(filepath, group="navigation_data") as nav_ds:
            # Merge navigation coords if needed
            nav_ds = nav_ds.set_coords(("longitude", "latitude"))
            ds = xr.merge([geo_ds, nav_ds.coords])
            # Assume reflectance variable is named "Rrs" with dims e.g. ("wavelength_3d", "y", "x") or ("wavelength_3d", "latitude", "longitude").
            rrs = ds["Rrs"]  # DataArray
            rrs = rrs.assign_coords(wavelength_3d = wave_all)
            # The wavelength coordinate may be named e.g. "wavelength_3d" or "wavelength". Inspect:
            if "wavelength_3d" in rrs.coords:
                wl_coord = "wavelength_3d"
            elif "wavelength" in rrs.coords:
                wl_coord = "wavelength"
            else:
                raise ValueError("Cannot find wavelength coordinate in Rrs")
            wavelengths = rrs[wl_coord].values  # e.g. array([400, 412.5, ...])
    except Exception as e:
        raise RuntimeError(f"Failed to open or interpret PACE granule {filepath}: {e}")

    # 2. Regrid: You need a regrid function that can handle a 3D DataArray. Two approaches:
    #    a) Loop over wavelengths, regrid each 2D slice separately.
    #    b) If your regrid supports multi-dim interpolation, pass the full 3D array.
    # Here we illustrate option (a), assuming regrid_2d(arr2d, bbox, res) returns a 2D numpy array at target grid.
    # Suppose sensor_params["res_km"] or similar defines resolution; adapt as needed.
    regridded_slices = []
    for wl in wavelengths:
        # Select nearest wavelength slice
        try:
            slice_da = rrs.sel({wl_coord: wl}, method="nearest")  # yields 2D DataArray with dims like ("y", "x") or ("latitude", "longitude")
        except Exception:
            continue  # or log warning
        # Now slice_da has coordinates latitude/longitude arrays. Convert to numpy and regrid:

        lat_arr = ds["latitude"].values   # 2D or 1D broadcastable
        lon_arr = ds["longitude"].values
        result = regrid_pace_slice(slice_da, lat_arr, lon_arr, bbox, sensor_params["res_km"])
        if result is None:
            logging.warning("Regrid PACE slice failed.")
            # mark processed, clean up, and return
            break
        regridded_2d, target_lats, target_lons = result
        regridded_slices.append((wl, regridded_2d))

    if not regridded_slices:
        return None  # no valid bands
    # Stack into an xarray DataArray or numpy array: shape (n_wl, ny, nx)
    wls, arrs = zip(*regridded_slices)
    arr_stack = np.stack(arrs, axis=0)
    # Optionally wrap into DataArray:
    # coords for target grid: 
    #   lat_centers, lon_centers computed elsewhere; or build from regrid output
    # da = xr.DataArray(arr_stack, coords={"wavelength": wls, "y": ..., "x": ...}, dims=("wavelength","y","x"))
    return wls, arr_stack, target_lats, target_lons  # wavelengths array and 3D numpy array

# Helper to extract datetime from filename
def extract_datetime_from_filename(path):
    filename = os.path.basename(path)
    m = re.search(r"(\d{8}T\d{6})", filename)
    if not m:
        logging.warning(f"No timestamp pattern in filename: {filename}")
        return None
    ts = m.group(1)
    try:
        return datetime.strptime(ts, "%Y%m%dT%H%M%S")
    except Exception as e:
        logging.error(f"Failed to parse timestamp '{ts}' in {filename}: {e}")
        return None

# Estimate position by linear interp/extrap
def estimate_position(times, lat_arr, lon_arr, t0):
    """
    Given:
      - times: list of pandas.Timestamp of station obs
      - lat_arr, lon_arr: either 0-D scalars or 1-D arrays of same length as times
      - t0: target Timestamp
    Returns:
      (lat0, lon0) interpolated (or fallback) position at t0.
    """
    # -- ensure these are at least 1-D numpy arrays --
    lat_arr = np.atleast_1d(lat_arr)
    lon_arr = np.atleast_1d(lon_arr)

    # if there's only one observation, just return it
    if lat_arr.size == 1 or lon_arr.size == 1:
        return float(lat_arr[0]), float(lon_arr[0])

    # Otherwise do your normal interpolation or nearest‐neighbor logic.
    # For example, if times is sorted and you just pick the closest time:
    #   find idx = argmin(|times[i] - t0|)
    #   return lat_arr[idx], lon_arr[idx]
    deltas = [abs((ts - t0).total_seconds()) for ts in times]
    i = int(np.argmin(deltas))
    return float(lat_arr[i]), float(lon_arr[i])

def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

def load_station_colors(path_json):
    if os.path.exists(path_json):
        with open(path_json, "r") as f:
            return json.load(f)
    else:
        return {}

def save_station_colors(path_json, station_colors):
    with open(path_json, "w") as f:
        json.dump(station_colors, f, indent=2)

def assign_color_for_station(station_name, station_colors, color_cycle):
    if station_name in station_colors:
        return station_colors[station_name]
    else:
        color = color_cycle[len(station_colors) % len(color_cycle)]
        station_colors[station_name] = color
        return color
    
def get_granule_filename(item):
    # Try common attribute names:
    urls = None
    if hasattr(item, "data"):
        urls = item.data
    elif hasattr(item, "urls"):
        urls = item.urls
    elif hasattr(item, "get_data_urls"):
        try:
            urls = item.get_data_urls()
        except:
            urls = None
    # If still None, you can fallback to parsing repr (less robust):
    if urls is None:
        txt = repr(item)
        import re
        m = re.search(r"Data:\s*\[\s*['\"](https?://[^'\"]+)['\"]", txt)
        if m:
            urls = [m.group(1)]
    if not urls:
        return None
    # Choose the first URL (or last if you prefer):
    url = urls[0]
    # Extract filename
    return os.path.basename(url)

def regrid_granule(dataset, bbox, res_km):
    lon_min, lat_min, lon_max, lat_max = bbox
    lat0 = (lat_min + lat_max) / 2.0
    res_lat_deg = res_km / 111.0
    res_lon_deg = res_km / (111.0 * math.cos(math.radians(lat0)))
    target_lats = np.arange(lat_min, lat_max + 1e-6, res_lat_deg)
    target_lons = np.arange(lon_min, lon_max + 1e-6, res_lon_deg)
    lon2d, lat2d = np.meshgrid(target_lons, target_lats)
    area_def = geometry.GridDefinition(lons=lon2d, lats=lat2d)

    lons = dataset["longitude"].values.flatten()
    lats = dataset["latitude"].values.flatten()

    regridded = {}
    bands = [name for name in dataset.data_vars if name.startswith("Rrs_")]
    if not bands:
        logging.warning("No Rrs_ bands in this granule")
        return None
    mask = (
        (lons >= lon_min - 1.0) & (lons <= lon_max + 1.0) &
        (lats >= lat_min - 1.0) & (lats <= lat_max + 1.0)
    )
    for band in bands:
        logging.info(f"Regridding band {band}")
        data = dataset[band].values.flatten()
        # Treat zeros as missing
        data[data == 0] = np.nan
        data_local = data[mask]
        lons_local = lons[mask]
        lats_local = lats[mask]
        valid = ~np.isnan(data_local) & ~np.isnan(lons_local) & ~np.isnan(lats_local)

        if not np.any(valid):
            logging.warning(f"No valid data for band {band} in bbox region")
            return None
        swath_def = geometry.SwathDefinition(lons=lons_local[valid], lats=lats_local[valid])
        try:
            radius_m = res_km * 1000
            result = kd_tree.resample_nearest(
                swath_def, data_local[valid], area_def,
                radius_of_influence=radius_m,
                fill_value=np.nan
            )
        except Exception as e:
            logging.error(f"Resampling failed for band {band}: {e}")
            return None
        da = xr.DataArray(
            result,
            dims=("latitude", "longitude"),
            coords={"latitude": target_lats, "longitude": target_lons},
            name=band
        )
        regridded[band] = da
    return regridded

def regrid_pace_slice(slice_da, lat_arr, lon_arr, bbox, res_km):
    """
    Regrid one 2D DataArray (slice_da) of reflectance at a single wavelength
    onto the target lat/lon grid defined by bbox and res_km.
    - slice_da: xarray.DataArray with dims like ("y","x") or ("latitude","longitude"),
      containing the Rrs values for one wavelength.
    - lat_arr, lon_arr: numpy arrays of same shape as slice_da.values, giving lat/lon per pixel.
      E.g., ds["latitude"].values, ds["longitude"].values from navigation_data.
    - bbox: (lon_min, lat_min, lon_max, lat_max).
    - res_km: target resolution in km.
    Returns: 2D numpy array of shape (n_lat, n_lon) on the target grid, with NaNs where no data.
    """
    lon_min, lat_min, lon_max, lat_max = bbox
    # center latitude for lon-degree scaling
    lat0 = (lat_min + lat_max) / 2.0
    res_lat_deg = res_km / 111.0
    res_lon_deg = res_km / (111.0 * math.cos(math.radians(lat0)))
    # build target grid centers

    target_lats = np.arange(lat_min, lat_max + 1e-6, res_lat_deg)
    target_lons = np.arange(lon_min, lon_max + 1e-6, res_lon_deg)

    lon2d, lat2d = np.meshgrid(target_lons, target_lats)
    area_def = geometry.GridDefinition(lons=lon2d, lats=lat2d)

    # Flatten the source arrays
    data = slice_da.values.flatten()
    lats = lat_arr.flatten()
    lons = lon_arr.flatten()

    # Mask to bounding box ± some margin
    mask = (
        (lons >= lon_min - 1.0) & (lons <= lon_max + 1.0) &
        (lats >= lat_min - 1.0) & (lats <= lat_max + 1.0)
    )
    data_local = data[mask]
    lats_local = lats[mask]
    lons_local = lons[mask]
    valid = np.isfinite(data_local) & np.isfinite(lats_local) & np.isfinite(lons_local)

    if not np.any(valid):
        logging.warning("No valid data for this wavelength slice in bbox region")
        return None

    swath_def = geometry.SwathDefinition(lons=lons_local[valid], lats=lats_local[valid])
    try:
        radius_m = res_km * 1000
        result = kd_tree.resample_nearest(
            swath_def, data_local[valid], area_def,
            radius_of_influence=radius_m,
            fill_value=np.nan
        )
    except Exception as e:
        logging.error(f"Resampling failed for wavelength slice: {e}")
        return None

    return result, target_lats, target_lons  # 2D numpy array on (target_lats, target_lons)

def extract_patch_from_regridded(regridded, lon0, lat0, pixel_count, res_km):
    half_km = (pixel_count * res_km) / 2.0
    half_lat_deg = half_km / 111.0
    half_lon_deg = half_km / (111.0 * math.cos(math.radians(lat0)))
    patch_arrays = {}
    total_cells = pixel_count * pixel_count
    for band, da in regridded.items():
        patch = da.sel(
            longitude=slice(lon0 - half_lon_deg, lon0 + half_lon_deg),
            latitude=slice(lat0 - half_lat_deg, lat0 + half_lat_deg)
        ).values
        patch = patch.astype(float)
        patch[patch == 0] = np.nan
        h0, w0 = patch.shape
        tgt = pixel_count
        def adjust(arr, tgt):
            h, w = arr.shape
            if h < tgt or w < tgt:
                new = np.full((tgt, tgt), np.nan, dtype=arr.dtype)
                si = max((tgt - h)//2, 0)
                sj = max((tgt - w)//2, 0)
                new[si:si+h, sj:sj+w] = arr
                return new
            elif h > tgt or w > tgt:
                si = (h - tgt)//2
                sj = (w - tgt)//2
                return arr[si:si+tgt, sj:sj+tgt]
            else:
                return arr
        patch_adj = adjust(patch, tgt)
        valid_count = np.count_nonzero(~np.isnan(patch_adj))
        coverage = valid_count / total_cells
        if coverage < 0.4:
            logging.info(f"Patch coverage for band {band} is {coverage:.2%} < 40% → skip station")
            return None, None, None
        patch_arrays[band] = patch_adj
    return patch_arrays, half_lon_deg, half_lat_deg

def plot_true_color(ax, regridded_dict, bbox, station_patches, station_colors):
    """
    ax: Cartopy GeoAxes (PlateCarree)
    regridded_dict: dict band_name -> 2D DataArray over full bbox grid
    bbox: (lon_min, lat_min, lon_max, lat_max)
    station_patches: list of dicts with station overlay info
    station_colors: dict station_name -> color string
    """
    # Identify available band wavelengths
    band_names = list(regridded_dict.keys())
    # Parse numeric wavelength from band names, e.g., "Rrs_667" → 667
    wavelengths = {}
    for b in band_names:
        parts = b.split("_")
        try:
            wl = float(parts[-1])
            wavelengths[b] = wl
        except:
            continue
    if len(wavelengths) < 3:
        logging.warning("Fewer than 3 numeric Rrs_ bands for true-color; skipping true-color plot.")
        return
    # Target wavelengths for RGB (nm)
    target = {"red": 667, "green": 555, "blue": 443}
    chosen = {}
    for color, tgt in target.items():
        # pick band with minimal abs(wl - tgt)
        b_sel = min(wavelengths.keys(), key=lambda b: abs(wavelengths[b] - tgt))
        chosen[color] = b_sel
    logging.info(f"True-color bands chosen: {chosen}")
    # Extract arrays
    arrs = {}
    for color, bname in chosen.items():
        da = regridded_dict.get(bname)
        if da is None:
            logging.warning(f"Band {bname} missing; skip true-color.")
            return
        arrs[color] = da.values

    normed = {}
    # Define per-channel percentiles and gains
    pct_params = {
        "red":   (0.02, 0.98, 1.0),
        "green": (0.02, 0.98, 1.0),  # boost green
        "blue":  (0.02, 0.98, 1.0)   # optionally reduce blue
    }
    for color, arr in arrs.items():
        flat = arr.flatten()
        flat = flat[~np.isnan(flat)]
        if flat.size == 0:
            logging.warning(f"No valid data in band {chosen[color]}; skip true-color.")
            return
        lowp, highp, gain = pct_params.get(color, (0.02, 0.98, 1.0))
        vmin = np.quantile(flat, lowp)
        vmax = np.quantile(flat, highp)
        logging.info(f"{color} ({chosen[color]}) stretch {lowp*100:.1f}–{highp*100:.1f}% → vmin={vmin:.4f}, vmax={vmax:.4f}, gain={gain}")
        norm = np.clip((arr - vmin) / (vmax - vmin), 0, 1)
        # gamma
        gamma = 1/2.2
        norm = np.clip(norm ** gamma, 0, 1)
        # apply gain
        normed[color] = np.clip(norm * gain, 0, 1)

    # Stack and plot as before
    rgb = np.stack([normed["red"], normed["green"], normed["blue"]], axis=-1)
    h, w, _ = rgb.shape
    rgba = np.zeros((h, w, 4), dtype=float)
    rgba[..., :3] = rgb
    # alpha = 0 where any channel is NaN, else 1
    mask_nan = np.isnan(rgb).any(axis=2)
    rgba[..., 3] = np.where(mask_nan, 0.0, 1.0)
    da0 = regridded_dict[chosen["red"]]
    lon0, lon1 = float(da0.longitude.min()), float(da0.longitude.max())
    lat0, lat1 = float(da0.latitude.min()), float(da0.latitude.max())
    extent = [lon0, lon1, lat0, lat1]
    ax.imshow(
        rgba,
        origin='lower',
        extent=extent,
        transform=ccrs.PlateCarree()
    )
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.coastlines(resolution='10m', linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.STATES, linewidth=0.5)
    gl = ax.gridlines(draw_labels=True, linewidth=0.2, color='gray', alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()

