import os
import numpy as np
import json
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict

def standardize_spatial(cube: np.ndarray, target_shape: Tuple[int,int]) -> np.ndarray:
    """
    Clip or drop rows/cols to match target_shape (nlat, nlon).
    Prioritize dropping edges with all NaNs.
    """
    arr = cube.copy()
    nlat, nlon, nb = arr.shape
    # Clip rows
    while nlat > target_shape[0]:
        # check first and last row for all NaN
        if np.isnan(arr[0,:,:]).all():
            arr = arr[1:,:,:]
        elif np.isnan(arr[-1,:,:]).all():
            arr = arr[:-1,:,:]
        else:
            # no all-NaN rows, drop last
            arr = arr[:-1,:,:]
        nlat = arr.shape[0]
    # Clip cols
    while nlon > target_shape[1]:
        if np.isnan(arr[:,0,:]).all():
            arr = arr[:,1:,:]
        elif np.isnan(arr[:,-1,:]).all():
            arr = arr[:,:-1,:]
        else:
            arr = arr[:,:-1,:]
        nlon = arr.shape[1]
    return arr

def load_observation_dataset(
    data_dir: str,
    spatial_shape: Tuple[int,int] = (10,10)
) -> Tuple[np.ndarray, List[int], np.ndarray, List[Dict]]:
    """
    Loads .npy files, standardizes spatial dims to spatial_shape by clipping,
    skips entirely NaN cubes, then prunes bands never valid across all entries.
    Returns:
      - dataset: structured array of shape (N,), dtype [('algae_class',int32),
                   ('Rrs_mean',float32,(10,10,n_bands_pruned))]
      - labels: list of int algae_class
      - wavelengths_pruned: 1D array of surviving wavelengths
    """  
    
    filepaths = sorted([
        os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".npy")
    ])

    cubes, classes, meta_list = [], [], []
    for path in filepaths:
        fname = os.path.basename(path)
        # parse metadata from filename: label_lat_lon_YYYYMMDD.npy
        parts = fname[:-4].split("_")
        _, lat, lon, date_str = parts
        # load cube
        d = np.load(path, allow_pickle=True).item()
        cube = d["Rrs_mean"]
        if not np.isfinite(cube).any():
            continue
        cube_std = standardize_spatial(cube, spatial_shape)
        cubes.append(cube_std.astype(np.float32))
        classes.append(int(d["algae_class"]))
        meta_list.append({
            "date": date_str,
            "lat": float(lat),
            "lon": float(lon)
        })

    if not cubes:
        raise ValueError("No valid cubes found after skipping all-NaN files.")
    # determine band pruning
    stacked = np.stack(cubes, axis=0)  # (N, h, w, b)
    # any valid at each band
    valid_per_band = np.isfinite(stacked).any(axis=(0,1,2))
    bands_idx = np.where(valid_per_band)[0]
    # prune wavelengths
    parent_dir = os.path.abspath(os.path.join(data_dir, os.pardir))
    wls_path = os.path.join(parent_dir, "combined_wavelengths.npy")
    if not os.path.exists(wls_path):
        raise FileNotFoundError(f"Cannot find wavelengths at {wls_path}")
    wls = np.load(wls_path)
    wavelengths_pruned = wls[bands_idx]
    # build structured array
    N = len(cubes)
    h, w, _ = cubes[0].shape
    b_pruned = len(bands_idx)
    dtype = np.dtype([
        ("algae_class", np.int32),
        ("Rrs_mean", np.float32, (h, w, b_pruned))
    ])
    result = np.zeros((N,), dtype=dtype)
    for i, cube in enumerate(cubes):
        pruned = cube[:,:,bands_idx]
        result[i]["algae_class"] = classes[i]
        result[i]["Rrs_mean"] = pruned
    return result, classes, wavelengths_pruned, meta_list


# ---------- Part 1: Save outputs to disk ----------------

# Paths
data_dir = "../LabelData/data/"
output_dir = "../LabelData/"
os.makedirs(output_dir, exist_ok=True)


# (Re)generate dataset, labels, wavelengths
dataset, labels, wavelengths, meta_list = load_observation_dataset(data_dir)

# save the metadata
with open(os.path.join(output_dir, "metadata.json"), "w") as f:
    json.dump(meta_list, f, indent=2)

# existing saves
np.save(os.path.join(output_dir, "dataset.npy"), dataset)
np.save(os.path.join(output_dir, "wavelengths_pruned.npy"), wavelengths)
with open(os.path.join(output_dir, "labels.json"), "w") as f:
    json.dump(labels, f)

print("Saved dataset.npy, wavelengths_pruned.npy, and labels.json")

# ---------- Part 2: Viewer ----------------
import os
import numpy as np
import json
import matplotlib.pyplot as plt

# Paths
dataset_path    = "../LabelData/dataset.npy"
wls_full_path   = "../LabelData/combined_wavelengths.npy"
label_map_path  = "../LabelData/label_to_species.json"
meta_path       = "../LabelData/metadata.json"

# Load everything
dataset      = np.load(dataset_path)
wls_full     = np.load(wls_full_path)
label_map    = json.load(open(label_map_path))
meta_list    = json.load(open(meta_path))

# Fixed RGB bands in the 302-band cube
bands_rgb = {"R": 141, "G": 93, "B": 66}
wls_pruned = np.load("../LabelData/wavelengths_pruned.npy")

print("num observations:", len(dataset))
print(dataset[0][1].shape)

for entry, meta in zip(dataset, meta_list):
    # 1) RGB composite from the **raw** full-band cube:
    #    We need to reload that raw cube:
    #    parse its filename from meta:
    fname = f"{entry['algae_class']}_{meta['lat']}_{meta['lon']}_{meta['date']}.npy"
    raw = np.load(os.path.join("../LabelData/data/", fname), allow_pickle=True).item()
    cube_full = raw["Rrs_mean"]

    # build rgb
    iR, iG, iB = bands_rgb["R"], bands_rgb["G"], bands_rgb["B"]
    rgb = np.stack([cube_full[..., iR],
                    cube_full[..., iG],
                    cube_full[..., iB]], axis=-1)
    nodata = np.all(np.isnan(rgb), axis=-1)
    rgb = np.nan_to_num(rgb, nan=0.0)
    mn, mx = rgb.min(), rgb.max()
    if mx>mn:
        rgb = (rgb-mn)/(mx-mn)
    rgb[nodata] = 0.0
    rgb = np.clip(rgb,0,1)

    # 2) Mean spectrum from the **pruned** arr
    arr_pruned = entry["Rrs_mean"]
    mean_spec = np.nanmean(
        arr_pruned.reshape(-1, arr_pruned.shape[2]), axis=0
    )

    # 3) Plot
    fig, (ax_img, ax_spec) = plt.subplots(1,2,figsize=(10,5))
    # Image
    ax_img.imshow(rgb, origin="upper")
    species = label_map[str(entry["algae_class"])]
    ax_img.set_title(f"{species}\n{meta['date']} @ ({meta['lat']:.3f},{meta['lon']:.3f})")
    ax_img.axis("off")
    # Spectrum

    ax_spec.plot(wls_pruned, mean_spec, "-o")
    ax_spec.set_xlabel("Wavelength (nm)")
    ax_spec.set_ylabel("Mean reflectance")
    ax_spec.grid(True)

    plt.tight_layout()
    plt.show(block=False)

    print(f"[{fname}] Press Enter to continue…", end="", flush=True)
    input()               # wait for user in console
    plt.close("all")
