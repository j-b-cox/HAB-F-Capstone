import os
import numpy as np
import matplotlib.pyplot as plt

data_dir = '../LabelData/data/'
wls = np.load("../LabelData/combined_wavelengths.npy")

# Only iterate over .npy files
files = sorted(f for f in os.listdir(data_dir) if f.endswith('.npy'))

# Fixed RGB band indices:
band_idx = {"R": 141, "G":  93, "B":  66}  # ◀ use these bands

for fname in files:
    sample_path = os.path.join(data_dir, fname)
    d = np.load(sample_path, allow_pickle=True).item()
    cube = d["Rrs_mean"]            # shape (nlat, nlon, n_bands)

    # 1) Count valid pixels per band (optional)
    # valid_counts = np.array([np.isfinite(cube[..., b]).sum()
    #                          for b in range(cube.shape[2])])

    # 2) (Optional) plot availability—skip if you don’t need it

    # 3) Build RGB composite from fixed bands
    iR, iG, iB = band_idx["R"], band_idx["G"], band_idx["B"]
    rgb = np.stack([cube[..., iR],
                    cube[..., iG],
                    cube[..., iB]], axis=-1)

    # 4) make nodata mask
    nodata_mask = np.all(np.isnan(rgb), axis=-1)

    # 5) NaN → 0, then normalize
    rgb = np.nan_to_num(rgb, nan=0.0, posinf=0.0, neginf=0.0)
    mn, mx = rgb.min(), rgb.max()
    if mx > mn:
        rgb = (rgb - mn)/(mx - mn)
    else:
        rgb[:] = 0.0

    # 6) force no-data pixels black
    rgb[nodata_mask] = 0.0

    # 7) display
    plt.figure(figsize=(6,6))
    plt.imshow(rgb, origin="upper")
    plt.title(f"{fname} — RGB @ bands [{iR},{iG},{iB}]")
    plt.axis("off")
    plt.show(block=False)

    # 8) click-through
    print("Press any key or click on figure to advance …")
    plt.waitforbuttonpress()
    plt.close('all')
