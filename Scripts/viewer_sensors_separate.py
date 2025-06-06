import matplotlib.pyplot as plt
import numpy as np
import pickle


def interactive_spectral_viewer(arr_5d, meta):
    """
    Interactive plot to click through days.
    For each day, plot spectral curves for each sensor averaged over pixels.
    """
    n_days, nlat, nlon, n_sensors, max_n_bands = arr_5d.shape
    wavelengths_list = meta['wavelengths']
    date_list = meta['date_list']
    sensors = meta['sensors']

    fig, ax = plt.subplots(figsize=(10,6))
    plt.subplots_adjust(bottom=0.2)

    day_idx = 0

    def plot_day(idx):
        ax.clear()
        for s_idx in range(n_sensors):
            data = arr_5d[idx, :, :, s_idx, :]  # shape (nlat, nlon, bands)
            # Flatten pixels, ignore NaNs
            pixels = data.reshape(-1, data.shape[-1])
            with np.errstate(invalid='ignore'):
                mean_spectrum = np.nanmean(pixels, axis=0)
            wave = wavelengths_list[s_idx]
            # wave may be shorter than max_n_bands, trim mean_spectrum accordingly
            mean_spectrum = mean_spectrum[:len(wave)]
            ax.plot(wave, mean_spectrum, label=sensors[s_idx])
        ax.set_title(f"Date: {date_list[idx]} (Day {idx+1} of {n_days})")
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Mean Reflectance")
        ax.legend()
        ax.grid(True)
        fig.canvas.draw_idle()

    def on_key(event):
        nonlocal day_idx
        if event.key == 'right':
            day_idx = (day_idx + 1) % n_days
            plot_day(day_idx)
        elif event.key == 'left':
            day_idx = (day_idx - 1) % n_days
            plot_day(day_idx)

    plot_day(day_idx)
    fig.canvas.mpl_connect('key_press_event', on_key)
    print("Use left/right arrow keys to navigate days.")
    plt.show()

data_path = "../Data/aggregated_separate_20240414-20240420.npy"
metadata_path = "../Data/aggregated_separate_20240414-20240420_metadata.pkl"

ndarray_all = np.load(data_path)  # shape: (n_days, nlat, nlon, nchannels)
print("shape:", ndarray_all.shape)

# Assume arr is your ndarray
nan_count = np.isnan(ndarray_all).sum()
print(f"Total NaNs: {nan_count}")

total_elements = ndarray_all.size
nan_percentage = 100 * nan_count / total_elements
print(f"NaN percentage: {nan_percentage:.2f}%")

# ---------------------------
# Load metadata
# ---------------------------
with open(metadata_path, "rb") as f:
    metadata = pickle.load(f)

interactive_spectral_viewer(ndarray_all, metadata)
