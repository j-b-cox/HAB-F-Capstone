#!/usr/bin/env python3
"""
labeler.py

Simplified labeling: for each observation, compute the average spectral curve
over all sensors and days (±4 days) within a bounding box around the point.
Uses per-granule caching of spectra to speed repeated runs.

Requires multi_sensor_aggregator to define:
  - get_reference_wavelengths(sensor, bbox)
  - load_daily_spectrum(date, sensor, bbox, wave_dict, cache_dir, combined_wavelengths)
  - aggregate_sensors_simple(start_date, end_date, bbox, sensors, data_dir, cache_dir, wave_dict, combined_wavelengths)
which implement per-granule processing and caching.

Usage:
    python labeler.py
"""

import os
import json
import math
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from multi_sensor_aggregator import (
    get_reference_wavelengths,
    load_daily_spectrum,
    aggregate_sensors_simple
)

# ---------------------------
# Logging setup
# ---------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def build_observation_array_mean_spectral_simple(
    obs_csv_path: str,
    sensors: list,
    km_radius: float = 0.5,
    data_dir: str = "../Data/",
    cache_dir: str = "../Cache/",
    output_dir: str = "../LabelData/"
) -> dict:
    """
    Streaming version: for each observation in obs_csv_path with abundance=="Elevated"
    and (spec_name=="Centric Diatom" OR phylum=="Cyanobacteria"), compute:
      - For each day in ±4-day window, average spectrum across sensors via aggregate_sensors_simple.
      - Then compute exponential-weighted mean across days.
    Saves each as .npy with {"algae_class": int, "Rrs_mean": 1D array}.
    Returns label_to_species mapping {0: "Centric Diatom", 1: "Cyanobacteria"}.
    """
    logger.info("Starting simplified labeling for %s", obs_csv_path)
    os.makedirs(output_dir, exist_ok=True)
    data_out = os.path.join(output_dir, "data")
    os.makedirs(data_out, exist_ok=True)

    # 1. Load CSV, parse datetime, filter abundance == "Elevated"
    logger.info("Loading observations CSV")
    df = pd.read_csv(obs_csv_path, parse_dates=["datetime"])
    df = df[df["abundance"] == "Elevated"].reset_index(drop=True)
    logger.info("Filtered to Elevated abundance: %d rows", len(df))

    # 2. Filter to Centric Diatom vs Cyanobacteria
    if "spec_name" not in df.columns or "phylum" not in df.columns:
        raise ValueError("CSV must contain 'spec_name' and 'phylum' columns")
    mask = (df["spec_name"] == "Centric Diatom") | (df["phylum"] == "Cyanobacteria")
    df = df[mask].reset_index(drop=True)
    logger.info("Filtered to Centric Diatom vs Cyanobacteria: %d rows", len(df))

    # 3. Shuffle for robustness
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    logger.info("Shuffled dataframe for robustness")

    # 4. Assign labels: 0 = Centric Diatom, 1 = Cyanobacteria
    def assign_label(row):
        return 0 if row["spec_name"] == "Centric Diatom" else 1
    df["algae_class"] = df.apply(assign_label, axis=1)
    label_to_species = {0: "Centric Diatom", 1: "Cyanobacteria"}
    # Save mapping
    mapping_path = os.path.join(output_dir, "label_to_species.json")
    with open(mapping_path, "w") as f:
        json.dump(label_to_species, f, indent=2)
    logger.info("Saved label_to_species mapping to %s", mapping_path)

    # 5. Load or init registry
    registry_path = os.path.join(output_dir, "registry.csv")
    if os.path.exists(registry_path):
        registry_df = pd.read_csv(registry_path, parse_dates=["datetime"])
        logger.info("Loaded existing registry with %d entries", len(registry_df))
    else:
        registry_df = pd.DataFrame(columns=["spec_name", "latitude", "longitude", "datetime"])
        logger.info("Initialized new empty registry")

    # 6. Precompute (and cache) wavelength dict via get_reference_wavelengths
    #    Use a subfolder under output_dir to cache per-sensor wavelengths.
    bbox_all = (-83.62, 41.34, -82, 42.27)  # adjust for your region
    wave_dict: dict = {}
    wave_cache_dir = os.path.join(output_dir, "wavelength_cache")
    os.makedirs(wave_cache_dir, exist_ok=True)

    for sensor in sensors:
        cache_path = os.path.join(wave_cache_dir, f"wavelengths_{sensor}.npy")
        if os.path.exists(cache_path):
            try:
                wave = np.load(cache_path)
                wave_dict[sensor] = wave
                logger.info("Loaded cached wavelengths for %s", sensor)
                continue
            except Exception as e:
                logger.warning("Failed to load cached wavelengths for %s: %s. Recomputing.", sensor, e)
        # Compute and cache
        logger.info("Precomputing wavelengths for %s", sensor)
        wave = get_reference_wavelengths(sensor, bbox_all)
        wave = wave[wave <= 2300]
        wave_dict[sensor] = wave
        try:
            np.save(cache_path, wave)
            logger.info("Cached wavelengths for %s", sensor)
        except Exception as e:
            logger.warning("Could not cache wavelengths for %s: %s", sensor, e)
    logger.info("Completed loading/caching wavelengths")

    # 7. Compute combined_wavelengths once
    all_waves = np.unique(np.concatenate([wave_dict[s] for s in sensors]))
    combined_wavelengths = np.sort(all_waves)
    # Save combined wavelengths
    comb_path = os.path.join(output_dir, "combined_wavelengths.npy")
    if not os.path.exists(comb_path):
        np.save(comb_path, combined_wavelengths)
        logger.info("Saved combined wavelengths to %s", comb_path)

    # 8. Process each row
    for idx, row in df.iterrows():
        # Skip if already in registry (same spec_name, lat, lon, datetime)
        exists = False
        if not registry_df.empty:
            cond = (
                (registry_df["spec_name"] == row["spec_name"]) &
                (np.isclose(registry_df["latitude"], row["latitude"])) &
                (np.isclose(registry_df["longitude"], row["longitude"])) &
                (registry_df["datetime"] == row["datetime"])
            )
            if cond.any():
                exists = True
        if exists:
            logger.info("Skipping row %d: already processed", idx)
            continue

        lat0 = row["latitude"]
        lon0 = row["longitude"]
        target_date = pd.to_datetime(row["datetime"])

        # Build bounding box ± km_radius around (lat0, lon0)
        # Convert km_radius to degrees: approx 1 deg lat ≈111 km
        res_lat = km_radius / 111.0
        res_lon = km_radius / (111.0 * math.cos(math.radians(lat0)))
        # bbox: (lon_min, lat_min, lon_max, lat_max)
        bbox = (
            lon0 - res_lon,
            lat0 - res_lat,
            lon0 + res_lon,
            lat0 + res_lat
        )

        # Time window ±2 days as strings
        start_date_str = (target_date - timedelta(days=3)).strftime("%Y-%m-%d")
        end_date_str   = (target_date + timedelta(days=3)).strftime("%Y-%m-%d")

        logger.info("Processing row %d: %s at %s; bbox ±%.2f km", 
                    idx, row["spec_name"], row["datetime"], km_radius)

        # Call simplified aggregator: returns arr shape (n_days, n_bands)
        try:
            arr2d, meta = aggregate_sensors_simple(
                start_date=start_date_str,
                end_date=end_date_str,
                bbox=bbox,
                sensors=sensors,
                data_dir=data_dir,
                cache_dir=cache_dir,
                wave_dict=wave_dict,
                combined_wavelengths=combined_wavelengths
            )
        except Exception as e:
            logger.error("aggregate_sensors_simple failed for row %d: %s", idx, e, exc_info=True)
            continue

        # arr2d: shape (n_days, n_bands)
        dates = meta.get("dates", None)
        if dates is None:
            logger.error("meta['dates'] missing for row %d; skipping", idx)
            continue
        # Convert to pd.Timestamp
        dates = [pd.to_datetime(d) for d in dates]
        wavelengths = meta.get("wavelengths", combined_wavelengths)

        # Strip timezone if present:
        if isinstance(target_date, pd.Timestamp) and target_date.tzinfo is not None:
            target_date = target_date.tz_localize(None)
        # For dates list:
        dates = [
            (d.tz_localize(None) if isinstance(d, pd.Timestamp) and d.tzinfo is not None else d)
            for d in dates
        ]

        # Compute exponential weights over days
        weights = np.array([np.exp(-abs((d - target_date).days)) for d in dates], dtype=float)
        total_w = weights.sum()
        if total_w <= 0:
            logger.warning("Sum of weights zero for row %d; skipping", idx)
            continue
        weights /= total_w

        # Weighted mean across days
        n_days, n_bands = arr2d.shape
        weighted_sum = np.zeros(n_bands, dtype=float)
        weight_total = np.zeros(n_bands, dtype=float)
        for i in range(n_days):
            row_spec = arr2d[i]  # 1D array length n_bands
            finite = np.isfinite(row_spec)
            if not np.any(finite):
                continue
            w_i = weights[i]
            weighted_sum[finite] += w_i * row_spec[finite]
            weight_total[finite] += w_i
        spectrum = np.full(n_bands, np.nan, dtype=np.float32)
        valid = weight_total > 0
        spectrum[valid] = (weighted_sum[valid] / weight_total[valid]).astype(np.float32)

        # Save the final spectrum
        lat_r = round(lat0, 5)
        lon_r = round(lon0, 5)
        dt_str = target_date.strftime("%Y%m%d")
        class_id = int(row["algae_class"])
        fname = f"{class_id}_{lat_r}_{lon_r}_{dt_str}.npy"
        out_path = os.path.join(data_out, fname)
        np.save(out_path, {"algae_class": class_id, "Rrs_mean": spectrum})
        logger.info("Saved spectral file %s (bands=%d)", out_path, spectrum.size)

        # Append to registry
        registry_df.loc[len(registry_df)] = {
            "spec_name": row["spec_name"],
            "latitude": row["latitude"],
            "longitude": row["longitude"],
            "datetime": row["datetime"]
        }
        registry_df.to_csv(registry_path, index=False)
        logger.debug("Updated registry with entry %s", fname)

    logger.info("Completed labeling. Total entries in registry: %d", len(registry_df))
    return label_to_species


if __name__ == "__main__":
    obs_csv = "../PMN.csv"
    output_dir = "../LabelData/"
    label_map = build_observation_array_mean_spectral_simple(
        obs_csv_path=obs_csv,
        sensors=[
            "MODISA_L2_OC", "MODIST_L2_OC",
            "OLCIS3A_L2_EFR_OC", "OLCIS3B_L2_EFR_OC",
            "PACE_OCI_L2_AOP"
        ],
        km_radius=2.5,
        data_dir="../Data/",
        cache_dir="../Cache/",
        output_dir=output_dir
    )
    logger.info("Saved label map with %d classes.", len(label_map))
