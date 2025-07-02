#!/usr/bin/env python3
import os
import logging
from datetime import datetime, timezone
import numpy as np
import xarray as xr
import earthaccess
from tqdm import tqdm
from itertools import product

from helpers import (
    process_pace_granule,
    extract_pace_patch,
    extract_datetime_from_filename,
)

# ─── Configure logging ─────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ─── User parameters ───────────────────────────────────────────────────────────
START_DATE   = datetime(2024, 1,  1, 0, 0, 0, tzinfo=timezone.utc)
END_DATE     = datetime(2024, 12, 31, 23,59,59, tzinfo=timezone.utc)
BBOX         = (-83.5, 41.3, -82.45, 42.2)
SHORT_NAME   = "PACE_OCI_L2_AOP"
MAX_SAMPLES  = 50
PATCH_SIZE   = 5
BANDS        = 172

WL_REF_FILE  = "./data/PACE_OCI.20240603T180158.L2.OC_AOP.V3_0.nc"
OUT_MEANS    = "channel_means.npy"
OUT_STDS     = "channel_stds.npy"

# ─── Load wavelength reference ─────────────────────────────────────────────────
logging.info("Loading wavelength reference from %s", WL_REF_FILE)
wave_all = xr.open_dataset(WL_REF_FILE, group="sensor_band_parameters")["wavelength_3d"].data

# ─── Search for granules ───────────────────────────────────────────────────────
logging.info("Querying EarthAccess for %s granules between %s and %s",
             SHORT_NAME, START_DATE.isoformat(), END_DATE.isoformat())
items = earthaccess.search_data(
    short_name   = SHORT_NAME,
    temporal     = (START_DATE, END_DATE),
    bounding_box = BBOX
)
if not items:
    logging.error("No granules found for that range—exiting.")
    raise RuntimeError("No PACE AOP granules found for that range.")
logging.info("Found %d granules total", len(items))

# ─── Sort and sample granules ──────────────────────────────────────────────────
items.sort(key=lambda g: extract_datetime_from_filename(str(g)) or datetime.min)
step    = max(1, len(items) // MAX_SAMPLES)
sampled = [items[i] for i in range(0, len(items), step)][:MAX_SAMPLES]
logging.info("Sampling %d granules (every %dth)", len(sampled), step)

# ─── Extract patches and accumulate ────────────────────────────────────────────
patch_list = []
for gran in tqdm(sampled, desc="Granules", unit="granule"):
    url = str(gran)
    try:
        fileset = earthaccess.download([gran], './data/')
        f       = fileset[0]
    except Exception as e:
        logging.warning("Could not open %s: %s", url, e)
        continue

    # regrid
    result = process_pace_granule(
        f, BBOX, {"res_km": 1.2}, wave_all
    )
    if result is None:
        logging.warning(f"No valid PACE data in {f}; skipping.")
        # mark processed, clean up, and return
        continue
    wls, arr_stack, lat_c, lon_c = result

    # extract patches
    centers = list(product(lat_c, lon_c))
    for lat0, lon0 in tqdm(centers, desc=" Patches", leave=False, unit="patch"):
        pdict = extract_pace_patch(
            arr_stack, wls,
            lon0, lat0,
            PATCH_SIZE, lat_c, lon_c
        )
        band_stack = np.stack([pdict[wl] for wl in wls], axis=-1)
        mask       = np.any(~np.isnan(band_stack),
                            axis=-1, keepdims=True).astype("float32")
        p4         = np.concatenate([band_stack, mask], axis=-1)  # (5,5,173)
        patch_list.append(p4.reshape(-1, BANDS+1))

# ─── Compute and save stats ───────────────────────────────────────────────────
logging.info("Stacking %d patches → array shape", len(patch_list))
X_all = np.vstack(patch_list)  # shape (n_patches*25, 173)
logging.info("Computing channel means and stds from array of shape %s", X_all.shape)

channel_means = np.nanmean(X_all, axis=0)
channel_stds  = np.nanstd (X_all, axis=0)

np.save(OUT_MEANS, channel_means)
np.save(OUT_STDS,  channel_stds)
logging.info("Saved means → %s, stds → %s", OUT_MEANS, OUT_STDS)
