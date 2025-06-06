import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pickle
import csv
import os
from datetime import datetime, timedelta
import tkinter as tk
from tkinter import filedialog

# === CONFIGURATION ===
COMPOSITE_NPY = "../Images/composite_data.npy"
META_PKL = "../Images/composite_metadata.pkl"
LABEL_CSV = "../Images/algae_pixel_labels.csv"
START_DATE_STR = "2024-04-14"
RGB_WAVES = [645, 555, 450]  # Wavelengths for true-color

# === Load Data ===
data = np.load(COMPOSITE_NPY)
with open(META_PKL, "rb") as f:
    meta = pickle.load(f)

lat = meta["lat"]
lon = meta["lon"]
wavelengths = meta["wavelengths"]

# Get RGB indices
r_idx = np.argmin(np.abs(wavelengths - RGB_WAVES[0]))
g_idx = np.argmin(np.abs(wavelengths - RGB_WAVES[1]))
b_idx = np.argmin(np.abs(wavelengths - RGB_WAVES[2]))

# Date handling
start_date = datetime.strptime(START_DATE_STR, "%Y-%m-%d")
dates = [start_date + timedelta(days=i) for i in range(data.shape[0])]

# === Labels Dictionary ===
labels = {}  # Format: {(day_idx, y, x): 1}

# === Load Existing Labels ===
if os.path.exists(LABEL_CSV):
    with open(LABEL_CSV, "r") as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            day_idx, y, x = map(int, row[:3])
            labels[(day_idx, y, x)] = 1
    print(f"Loaded existing labels: {len(labels)} points.")

# === Normalization Helper ===
def normalize(arr, vmin=0, vmax=0.03):
    return np.clip((arr - vmin) / (vmax - vmin), 0, 1)

# === Interactive Plotting ===
class LabelingTool:
    def __init__(self):
        self.day_idx = 0
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.bboxes = []  # List of drawn boxes
        self.rects = []   # Rectangles on plot
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.kid = self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.drawing = False
        self.start = None
        self.lat_grid, self.lon_grid = np.meshgrid(lat, lon, indexing='ij')
        self.update_plot()

    def update_plot(self):
        self.ax.clear()
        rgb = normalize(data[self.day_idx, :, :, r_idx]), normalize(data[self.day_idx, :, :, g_idx]), normalize(data[self.day_idx, :, :, b_idx])
        rgb = np.stack(rgb, axis=-1)
        self.ax.imshow(rgb, origin='lower', extent=[lon.min(), lon.max(), lat.min(), lat.max()])
        self.ax.set_title(f"Date: {dates[self.day_idx].strftime('%Y-%m-%d')} (Use arrows to change day, R to remove last box, S to save)")
        self.ax.set_xlabel("Longitude")
        self.ax.set_ylabel("Latitude")
        self.ax.grid(True, alpha=0.3)
        for rect in self.rects:
            self.ax.add_patch(rect)
        self.fig.canvas.draw()

    def on_click(self, event):
        if event.inaxes != self.ax:
            return
        if not self.drawing:
            self.start = (event.xdata, event.ydata)
            self.drawing = True
        else:
            end = (event.xdata, event.ydata)
            x0, x1 = sorted([self.start[0], end[0]])
            y0, y1 = sorted([self.start[1], end[1]])
            rect = patches.Rectangle((x0, y0), x1 - x0, y1 - y0, linewidth=2, edgecolor='red', facecolor='none')
            self.rects.append(rect)
            self.bboxes.append((x0, x1, y0, y1))
            self.drawing = False
            self.update_plot()

    def on_key(self, event):
        if event.key == 'right':
            self.day_idx = (self.day_idx + 1) % data.shape[0]
            self.bboxes = []
            self.rects = []
            self.update_plot()
        elif event.key == 'left':
            self.day_idx = (self.day_idx - 1) % data.shape[0]
            self.bboxes = []
            self.rects = []
            self.update_plot()
        elif event.key == 'r':
            if self.bboxes:
                self.bboxes.pop()
                self.rects.pop()
                self.update_plot()
        elif event.key == 's':
            self.convert_boxes_to_labels()
            self.save_labels()
            print(f"Saved labels. Total points: {len(labels)}")

    def convert_boxes_to_labels(self):
        for bbox in self.bboxes:
            x0, x1, y0, y1 = bbox
            mask = (lon >= x0) & (lon <= x1) & (lat[:, None] >= y0) & (lat[:, None] <= y1)
            ys, xs = np.where(mask)
            for y, x in zip(ys, xs):
                labels[(self.day_idx, y, x)] = 1
        print(f"Labeled {len(self.bboxes)} box(es) for day {self.day_idx}.")
        self.bboxes = []
        self.rects = []

    def save_labels(self):
        with open(LABEL_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["day_idx", "y", "x", "label"])
            for (day_idx, y, x), label_val in labels.items():
                writer.writerow([day_idx, y, x, label_val])

# === Run Tool ===
tool = LabelingTool()
plt.show()
