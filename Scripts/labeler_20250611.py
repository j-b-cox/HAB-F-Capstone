import os
import math
import json
import numpy as np
import pandas as pd
import logging
from datetime import timedelta
from typing import Tuple, List, Dict
from multi_sensor_aggregator import aggregate_sensors, get_reference_wavelengths

# ---------------------------
# Logging setup
# ---------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def standardize_cube(arr: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
    """
    Pad or trim a 3D array `arr` to shape `target_shape`.
    Pads with np.nan if arr is smaller, or trims if larger.
    """
    out = np.full(target_shape, np.nan, dtype=arr.dtype)
    min0 = min(arr.shape[0], target_shape[0])
    min1 = min(arr.shape[1], target_shape[1])
    min2 = min(arr.shape[2], target_shape[2])
    out[:min0, :min1, :min2] = arr[:min0, :min1, :min2]
    return out


def build_observation_array_mean_streaming(
    obs_csv_path: str,
    sensors: List[str],
    n_top: int = 10,
    resolution: float = 0.005,
    data_dir: str = "../Data/",
    cache_dir: str = "../Cache/",
    output_dir: str = "../LabelData/"
) -> Dict[int, str]:
    """
    Streaming version: processes observations one at a time, saves each result as soon as it's created,
    and skips already-computed entries. Returns label_to_species dict.

    Saves:
      - structured data array (1 record per .npy file):  output_dir/data/
      - label→species mapping as JSON:                  output_dir/label_to_species.json
      - registry of completed entries:                  output_dir/registry.csv
    """
    logger.info("Starting streaming labeling for %s", obs_csv_path)
    os.makedirs(output_dir, exist_ok=True)
    data_out = os.path.join(output_dir, "data")
    os.makedirs(data_out, exist_ok=True)

    # 1. Load CSV, parse datetime, and filter to 'Elevated'
    logger.info("Loading observations CSV")
    df = pd.read_csv(obs_csv_path, parse_dates=["datetime"] )
    df = df[df["abundance"] == "Elevated"].reset_index(drop=True)
    logger.info("Filtered to Elevated abundance: %d rows", len(df))

    # 2. Top-n species and filter
    top_species = df["spec_name"].value_counts().nlargest(n_top).index.tolist()
    df = df[df["spec_name"].isin(top_species)].reset_index(drop=True)
    logger.info("Selected top %d species: %s", n_top, top_species)

    # 3. Shuffle for robustness
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    logger.info("Shuffled dataframe for robustness")

    # 4. Integer label for each species
    df["algae_class"] = pd.Categorical(df["spec_name"], categories=top_species).codes
    label_to_species = {int(code): name for code, name in enumerate(top_species)}
    # 5. Save mapping
    mapping_path = os.path.join(output_dir, "label_to_species.json")
    with open(mapping_path, "w") as f:
        json.dump(label_to_species, f, indent=2)
    logger.info("Saved label_to_species mapping to %s", mapping_path)

    # 6. Load registry (or create empty one)
    registry_path = os.path.join(output_dir, "registry.csv")
    if os.path.exists(registry_path):
        registry_df = pd.read_csv(registry_path, parse_dates=["datetime"] )
        logger.info("Loaded existing registry with %d entries", len(registry_df))
    else:
        registry_df = pd.DataFrame(columns=["spec_name", "latitude", "longitude", "datetime"])
        logger.info("Initialized new empty registry")

    # 7. Precompute wavelength dict
    bbox_all = (-83.62, 41.34, -82, 42.27)
    wave_dict = {}
    for sensor in sensors:
        logger.info("Precomputing wavelengths for %s", sensor)
        wave = get_reference_wavelengths(sensor, bbox_all)
        wave_dict[sensor] = wave[wave <= 2300]
    logger.info("Completed precomputing wavelengths")

    standard_shape = None

    # 8. Process each row
    for idx, row in df.iterrows():
        key = (row["spec_name"], row["latitude"], row["longitude"], row["datetime"])
        # 8a. Skip if already in registry
        if ((registry_df["spec_name"] == key[0]) &
            (registry_df["latitude"] == key[1]) &
            (registry_df["longitude"] == key[2]) &
            (registry_df["datetime"] == pd.Timestamp(key[3]))).any():
            logger.info("Skipping already processed entry %s", key)
            continue

        # 8b. Build bounding box
        delta_lat = 2.5 / 111.0
        delta_lon = 2.5 / (111.0 * math.cos(math.radians(row["latitude"])))
        bbox = (
            row["longitude"] - delta_lon,
            row["latitude"] - delta_lat,
            row["longitude"] + delta_lon,
            row["latitude"] + delta_lat
        )

        # 8c. Time range
        start_date = (row["datetime"] - timedelta(days=2)).strftime("%Y-%m-%d")
        end_date   = (row["datetime"] + timedelta(days=2)).strftime("%Y-%m-%d")

        logger.info("Processing row %d: %s at %s", idx, key[0], key[3])
        try:
            # 8d. Aggregate and average
            arr4d, meta = aggregate_sensors(
                start_date=start_date,
                end_date=end_date,
                bbox=bbox,
                sensors=sensors,
                resolution=resolution,
                data_dir=data_dir,
                cache_dir=cache_dir,
                wave_dict=wave_dict
            )
            arr_mean = np.nanmean(arr4d, axis=0).astype(np.float32)

            # — save the true wavelength array (once) —
            wls_path = os.path.join(output_dir, "combined_wavelengths.npy")
            if not os.path.exists(wls_path):
                np.save(wls_path, meta["wavelengths"])
                logger.info(f"Saved combined wavelengths to {wls_path}")

            # --- standardize shape ---
            logger.info("Checking standard shape")
            if standard_shape is None:
                standard_shape = arr_mean.shape
                logger.info("Standard shape set to %s", standard_shape)
            else:
                arr_mean = standardize_cube(arr_mean, standard_shape)

        except Exception as e:
            logger.error("Error processing row %d: %s", idx, e)
            continue

        # 8e. Save to unique filename
        lat = round(row["latitude"], 5)
        lon = round(row["longitude"], 5)
        dt_str = pd.to_datetime(row["datetime"]).strftime("%Y%m%d")
        class_id = int(row["algae_class"])
        fname = f"{class_id}_{lat}_{lon}_{dt_str}.npy"
        out_path = os.path.join(data_out, fname)
        np.save(out_path, {"algae_class": class_id, "Rrs_mean": arr_mean})
        logger.info("Saved data file %s", out_path)

        # 8f. Append to registry
        registry_df.loc[len(registry_df)] = {
            "spec_name": row["spec_name"],
            "latitude": row["latitude"],
            "longitude": row["longitude"],
            "datetime": row["datetime"]
        }
        registry_df.to_csv(registry_path, index=False)
        logger.debug("Updated registry with entry %s", key)

    logger.info("Completed streaming labeling. Total entries: %d", len(registry_df))
    return label_to_species


if __name__ == "__main__":
    obs_csv = "../Data/PMN_20250605.csv"
    output_dir = "../LabelData/"
    label_map = build_observation_array_mean_streaming(
        obs_csv_path=obs_csv,
        sensors=[
            "MODISA_L2_OC", "MODIST_L2_OC",
            "OLCIS3A_L2_EFR_OC", "OLCIS3B_L2_EFR_OC",
            "PACE_OCI_L2_AOP"
        ],
        n_top=5,
        resolution=0.005,
        data_dir="../Data/",
        cache_dir="../Cache/",
        output_dir=output_dir
    )
    logger.info("Saved label map with %d entries.", len(label_map))
