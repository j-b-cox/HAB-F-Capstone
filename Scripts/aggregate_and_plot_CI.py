from multi_sensor_aggregator import aggregate_sensors
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from datetime import datetime, timedelta

START_DATE = "2024-08-14"
END_DATE   = "2024-08-20"
BBOX       = (-83.62, 41.34, -82, 42.27)
SENSORS    = ["MODISA_L2_OC", "MODIST_L2_OC",
              "OLCIS3A_L2_EFR_OC", "OLCIS3B_L2_EFR_OC",
              "PACE_OCI_L2_AOP"]
RES = 0.01

ndarray_all, meta = aggregate_sensors(
    start_date=START_DATE,
    end_date=END_DATE,
    bbox=BBOX,
    sensors=SENSORS,
    resolution=RES
)

wavelengths = meta["wavelengths"]
date_list = meta["date_list"]
n_days, nlat, nlon, nch = ndarray_all.shape

if date_list is None or len(date_list) != n_days:
    start_date = datetime(2024, 4, 14)
    date_list = [(start_date + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(n_days)]

λ1, λ2, λ3 = 665.0, 681.0, 709.0
i1 = int(np.argmin(np.abs(wavelengths - λ1)))
i2 = int(np.argmin(np.abs(wavelengths - λ2)))
i3 = int(np.argmin(np.abs(wavelengths - λ3)))

def substitute_nan_band(band_cube, target_idx, all_waves):
    band = band_cube[:, :, target_idx]
    nan_mask = np.isnan(band)
    if not np.any(nan_mask):
        return band
    out = band.copy()
    for offset in range(1, len(all_waves)):
        for direction in [-1, 1]:
            alt_idx = target_idx + offset * direction
            if 0 <= alt_idx < len(all_waves):
                alt_band = band_cube[:, :, alt_idx]
                replace_mask = nan_mask & ~np.isnan(alt_band)
                out[replace_mask] = alt_band[replace_mask]
                nan_mask = np.isnan(out)
                if not np.any(nan_mask):
                    return out
    return out

SS_stack = np.full((n_days, nlat, nlon), np.nan, dtype=np.float32)
for d in range(n_days):
    day_cube = ndarray_all[d]
    band1 = substitute_nan_band(day_cube, i1, wavelengths)
    band2 = substitute_nan_band(day_cube, i2, wavelengths)
    band3 = substitute_nan_band(day_cube, i3, wavelengths)
    SS = band2 - band1 + ((band3 - band1) * 0.3636)
    zero_mask = (band1 == 0) & (band2 == 0) & (band3 == 0)
    SS[zero_mask] = np.nan
    SS_stack[d] = SS

SS_flipped = np.flipud(SS_stack.transpose(1, 0, 2)).transpose(1, 0, 2)
data_flipped = np.flipud(ndarray_all.transpose(1, 0, 2, 3)).transpose(1, 0, 2, 3)

class SSViewer:
    def __init__(self, ss_images, full_data, wavelengths, dates):
        self.ss_images = ss_images
        self.full_data = full_data
        self.wavelengths = wavelengths
        self.dates = dates
        self.n_images = ss_images.shape[0]
        self.index = 0

        self.fig, (self.ax_img, self.ax_spec) = plt.subplots(
            2, 1, figsize=(8, 8), gridspec_kw={"height_ratios": [3, 1]}
        )

        cmap = plt.get_cmap("inferno").copy()
        cmap.set_bad(color=(0, 0, 0, 0))

        self.img_display = self.ax_img.imshow(
            np.ma.masked_invalid(self.ss_images[self.index]),
            cmap=cmap, origin="upper"
        )
        self.colorbar = self.fig.colorbar(self.img_display, ax=self.ax_img, fraction=0.03, pad=0.04)
        self.ax_img.set_title(f"{self.dates[self.index]} ({self.index+1}/{self.n_images})")
        self.ax_img.axis("off")

        self.text_info = self.ax_img.text(
            0.02, 0.95, "", color="white", fontsize=9,
            transform=self.ax_img.transAxes, va="top", ha="left",
            bbox=dict(facecolor="black", alpha=0.5, boxstyle="round,pad=0.3")
        )

        self.line, = self.ax_spec.plot([], [], "-o", markersize=4)
        self.ax_spec.set_xlabel("Wavelength (nm)")
        self.ax_spec.set_ylabel("Rrs")
        self.ax_spec.set_title("Pixel Spectrum")
        self.ax_spec.grid(True)

        axprev = plt.axes([0.7, 0.01, 0.1, 0.05])
        axnext = plt.axes([0.81, 0.01, 0.1, 0.05])
        self.bnext = Button(axnext, "Next")
        self.bprev = Button(axprev, "Previous")
        self.bnext.on_clicked(self.next)
        self.bprev.on_clicked(self.prev)

        self.cid = self.fig.canvas.mpl_connect("motion_notify_event", self.on_hover)
        plt.tight_layout()
        plt.show()

    def on_hover(self, event):
        if event.inaxes == self.ax_img and event.xdata is not None and event.ydata is not None:
            j, i = int(event.xdata + 0.5), int(event.ydata + 0.5)
            if 0 <= i < self.ss_images.shape[1] and 0 <= j < self.ss_images.shape[2]:
                ss_val = self.ss_images[self.index, i, j]
                if np.isfinite(ss_val):
                    self.text_info.set_text(f"Pixel ({i}, {j}): SS = {ss_val:.4f}")
                else:
                    self.text_info.set_text(f"Pixel ({i}, {j}): No Data")

                spectrum = self.full_data[self.index, i, j]
                self.ax_spec.clear()
                self.ax_spec.plot(self.wavelengths, spectrum, "-o", markersize=4)
                for wl, val in zip(self.wavelengths, spectrum):
                    if np.isfinite(val):
                        self.ax_spec.annotate(f"{int(wl)}", (wl, val), textcoords="offset points",
                                              xytext=(0, 4), ha='center', fontsize=8)
                self.ax_spec.set_xlabel("Wavelength (nm)")
                self.ax_spec.set_ylabel("Rrs")
                self.ax_spec.set_title("Pixel Spectrum")
                self.ax_spec.grid(True)
                self.fig.canvas.draw_idle()

    def update(self):
        data = np.ma.masked_invalid(self.ss_images[self.index])
        self.img_display.set_data(data)
        self.img_display.set_clim(vmin=np.nanmin(self.ss_images), vmax=np.nanmax(self.ss_images))
        self.ax_img.set_title(f"{self.dates[self.index]} ({self.index+1}/{self.n_images})")
        self.text_info.set_text("")
        self.ax_spec.clear()
        self.ax_spec.set_title("Hover to see pixel spectrum")
        self.ax_spec.set_xlabel("Wavelength (nm)")
        self.ax_spec.set_ylabel("Rrs")
        self.ax_spec.grid(True)
        self.fig.canvas.draw_idle()

    def next(self, event):
        self.index = (self.index + 1) % self.n_images
        self.update()

    def prev(self, event):
        self.index = (self.index - 1) % self.n_images
        self.update()

viewer = SSViewer(SS_flipped, data_flipped, wavelengths, date_list)
