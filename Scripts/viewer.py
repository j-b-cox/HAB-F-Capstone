import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from datetime import datetime, timedelta
from scipy.ndimage import generic_filter

# ---------------------------
# Load the aggregated data
# ---------------------------
data_path = "../Data/aggregated_20240414-20240420.npy"
metadata_path = "../Data/aggregated_20240414-20240420_metadata.pkl"

ndarray_all = np.load(data_path)  # shape: (n_days, nlat, nlon, nchannels)
print("shape:", ndarray_all.shape)
# ---------------------------
# Load metadata
# ---------------------------
with open(metadata_path, "rb") as f:
    metadata = pickle.load(f)

wavelengths = metadata["wavelengths"]   # 1D array of combined wavelengths
date_list = metadata["date_list"]       # list of "YYYY-MM-DD" strings

print("wavelengths.shape:", wavelengths.shape)
print("date_list:", date_list)

# If date_list is missing or mismatched, regenerate from ndarray length
n_days, nlat, nlon, nch = ndarray_all.shape
if date_list is None or len(date_list) != n_days:
    start_date = datetime(2024, 4, 14)  # adjust if needed
    date_list = [(start_date + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(n_days)]

# ---------------------------
# Define function to pick fallback RGB indices
# ---------------------------
def pick_rgb_indices(wavelengths: np.ndarray, data_day: np.ndarray) -> tuple:
    """
    For a single day’s 3D data_day (nlat, nlon, nchannels), attempt to pick
    R=667 nm, G=547 nm, B=488 nm. If any chosen band is entirely NaN, fall back
    to the nearest wavelength that has at least one valid pixel.
    """
    targets = [667.0, 547.0, 488.0]
    indices = [int(np.argmin(np.abs(wavelengths - t))) for t in targets]

    for idx_pos, idx in enumerate(indices):
        band = data_day[:, :, idx]
        if np.all(np.isnan(band)):
            diffs = np.abs(wavelengths - targets[idx_pos])
            order = np.argsort(diffs)
            for candidate in order:
                if np.any(np.isfinite(data_day[:, :, candidate])):
                    indices[idx_pos] = int(candidate)
                    break
    return tuple(indices)

# ---------------------------
# Build RGB_data via per-day/fallback logic
# ---------------------------
RGB_data = np.full((n_days, nlat, nlon, 3), np.nan, dtype=np.float32)

for d in range(n_days):
    day_slice = ndarray_all[d, :, :, :]  # (nlat, nlon, nchannels)
    Ri, Gi, Bi = pick_rgb_indices(wavelengths, day_slice)
    RGB_data[d, :, :, 0] = day_slice[:, :, Ri]
    RGB_data[d, :, :, 1] = day_slice[:, :, Gi]
    RGB_data[d, :, :, 2] = day_slice[:, :, Bi]

# ---------------------------
# Interpolate missing pixels (4-neighbor average)
# ---------------------------
def interpolate_nans_4neighbors(img: np.ndarray) -> np.ndarray:
    """
    Fill NaNs in a (nlat, nlon, 3) RGB image by averaging available
    up/down/left/right neighbors. If fewer than 1 valid neighbors, leave NaN.
    """
    def interpolate_fn(neighborhood):
        center = neighborhood[4]
        if not np.isnan(center):
            return center
        neighbors = [neighborhood[1], neighborhood[3], neighborhood[5], neighborhood[7]]
        valid_neighbors = [v for v in neighbors if not np.isnan(v)]
        return np.mean(valid_neighbors) if valid_neighbors else np.nan

    filled = np.zeros_like(img)
    for c in range(3):
        filled[..., c] = generic_filter(
            img[..., c],
            interpolate_fn,
            size=3,
            mode='constant',
            cval=np.nan
        )
    return filled

RGB_interpolated = np.array([
    interpolate_nans_4neighbors(RGB_data[d]) for d in range(n_days)
])

# ---------------------------
# Fill remaining NaNs by combining available channels
# ---------------------------
def fill_missing_rgb(img: np.ndarray) -> np.ndarray:
    """
    For any pixel where one of R, G, or B is still NaN after interpolation,
    fill it using the average of the other two channels (if available). If only
    one channel is valid, replicate it to the others to avoid zeroing.
    """
    out = img.copy()
    nlat, nlon, _ = img.shape
    for i in range(nlat):
        for j in range(nlon):
            r, g, b = out[i, j, :]
            valid = [np.isfinite(r), np.isfinite(g), np.isfinite(b)]
            vals = [r, g, b]
            # If all NaN, leave as is
            if not any(valid):
                continue
            # If exactly two are valid, fill the missing
            if sum(valid) == 2:
                missing_idx = valid.index(False)
                present_vals = [vals[k] for k in range(3) if valid[k]]
                out[i, j, missing_idx] = np.nanmean(present_vals)
            # If only one is valid, replicate it across all three
            elif sum(valid) == 1:
                present_idx = valid.index(True)
                out[i, j, :] = vals[present_idx]
    return out

RGB_filled = np.array([fill_missing_rgb(RGB_interpolated[d]) for d in range(n_days)])

# ---------------------------
# Normalize per image & apply γ=0.5
# ---------------------------
def normalize_and_gamma(img_rgb: np.ndarray) -> np.ndarray:
    """
    Given a (nlat, nlon, 3) array with possible NaNs:
      1. Compute a single 1st–99th percentile over all three channels.
      2. Clip & rescale to [0,1], turning NaNs into 0.
      3. Apply sqrt gamma (power 0.5).
    """
    flat = img_rgb.reshape(-1, 3)
    valid = flat[np.isfinite(flat)]
    if valid.size == 0:
        return np.zeros_like(img_rgb, dtype=np.float32)

    vmin = np.percentile(valid, 1)
    vmax = np.percentile(valid, 99)
    if np.isclose(vmax, vmin):
        clipped = np.clip(img_rgb, vmin, vmax)
        norm = clipped - vmin
        norm[np.isnan(norm)] = 0.0
        return np.sqrt(norm).astype(np.float32)

    clipped = np.clip(img_rgb, vmin, vmax)
    norm = (clipped - vmin) / (vmax - vmin)
    norm[np.isnan(norm)] = 0.0
    return np.sqrt(norm).astype(np.float32)

RGB_data_normalized = np.array([
    normalize_and_gamma(RGB_filled[d]) for d in range(n_days)
])

# ---------------------------
# Flip images vertically
# ---------------------------
RGB_data_flipped = np.flipud(RGB_data_normalized.transpose(1, 0, 2, 3)).transpose(1, 0, 2, 3)

# ---------------------------
# Viewer tool with inlaid spectral plot and pixel-hover RGB display
# ---------------------------
class ImageViewer:
    def __init__(self, images: np.ndarray, full_data: np.ndarray, wavelengths: np.ndarray, dates: list):
        """
        images: (n_days, nlat, nlon, 3) array with values in [0,1]
        full_data: (n_days, nlat, nlon, nchannels) original reflectance array
        wavelengths: 1D array of length nchannels
        dates: list of date strings ("YYYY-MM-DD"), length n_days
        """
        self.images = images
        self.full_data = full_data
        self.wavelengths = wavelengths
        self.dates = dates
        self.n_images = images.shape[0]
        self.index = 0

        # Create figure with two subplots: image on top, spectral on bottom
        self.fig, (self.ax_img, self.ax_spec) = plt.subplots(
            nrows=2,
            ncols=1,
            figsize=(8, 8),
            gridspec_kw={"height_ratios": [3, 1]}
        )

        # Display first image
        self.img_display = self.ax_img.imshow(self.images[self.index], origin="upper")
        self.ax_img.set_title(f"{self.dates[self.index]} ({self.index+1}/{self.n_images})")
        self.ax_img.axis("off")

        # Add a text annotation for pixel RGB values (initially empty)
        self.text_rgb = self.ax_img.text(
            0.02, 0.95, "", color="white", fontsize=9, 
            transform=self.ax_img.transAxes, va="top", ha="left",
            bbox=dict(facecolor="black", alpha=0.5, boxstyle="round,pad=0.3")
        )

        # Plot initial spectral curve
        self.plot_spectral_curve(self.index)

        # Buttons
        axprev = plt.axes([0.7, 0.01, 0.1, 0.05])
        axnext = plt.axes([0.81, 0.01, 0.1, 0.05])
        self.bnext = Button(axnext, "Next")
        self.bprev = Button(axprev, "Previous")
        self.bnext.on_clicked(self.next)
        self.bprev.on_clicked(self.prev)

        # Connect hover event
        self.cid = self.fig.canvas.mpl_connect("motion_notify_event", self.on_hover)

        plt.tight_layout()
        plt.show()

    def plot_spectral_curve(self, day_idx: int):
        """Plot mean reflectance vs. wavelength for the given day."""
        self.ax_spec.clear()
        data_day = self.full_data[day_idx, :, :, :]  # (nlat, nlon, nchannels)
        flat = data_day.reshape(-1, data_day.shape[-1])
        mean_vals = np.nanmean(flat, axis=0)
        self.ax_spec.plot(self.wavelengths, mean_vals, "-o", markersize=3)
        self.ax_spec.set_xlabel("Wavelength (nm)")
        self.ax_spec.set_ylabel("Mean Rrs")
        self.ax_spec.set_title("Spectral Curve")
        self.ax_spec.grid(True)

    def on_hover(self, event):
        """
        When the mouse moves over the image, display the RGB values
        of the underlying pixel in self.text_rgb.
        """
        if event.inaxes == self.ax_img:
            x, y = event.xdata, event.ydata
            if x is None or y is None:
                return

            # Convert float coords to integer pixel indices
            j = int(x + 0.5)
            i = int(y + 0.5)

            # Check bounds
            if 0 <= i < self.images.shape[1] and 0 <= j < self.images.shape[2]:
                rgb = self.images[self.index, i, j, :]  # normalized [0,1]
                # Convert to 0-255 for display
                r255 = int(np.clip(rgb[0] * 255, 0, 255))
                g255 = int(np.clip(rgb[1] * 255, 0, 255))
                b255 = int(np.clip(rgb[2] * 255, 0, 255))
                self.text_rgb.set_text(f"Pixel ({i}, {j}): R={r255}, G={g255}, B={b255}")
            else:
                self.text_rgb.set_text("")  # outside bounds
            self.fig.canvas.draw_idle()

    def update(self):
        """Refresh image, clear RGB text, and update spectral curve for current index."""
        self.img_display.set_data(self.images[self.index])
        self.ax_img.set_title(f"{self.dates[self.index]} ({self.index+1}/{self.n_images})")
        self.text_rgb.set_text("")  # clear on image change
        self.plot_spectral_curve(self.index)
        self.fig.canvas.draw_idle()

    def next(self, event):
        """Advance to next day."""
        self.index = (self.index + 1) % self.n_images
        self.update()

    def prev(self, event):
        """Go to previous day."""
        self.index = (self.index - 1) % self.n_images
        self.update()

# ---------------------------
# Run the viewer
# ---------------------------
viewer = ImageViewer(RGB_data_flipped, ndarray_all, wavelengths, date_list)
