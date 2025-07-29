#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import xarray as xr
from datetime import datetime, timedelta
import pickle
import time
import random
from tqdm import tqdm
import numpy as np
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import TimeDistributed, Conv2D, MaxPooling2D, Flatten, LSTM, Dense, Reshape, Input
from datetime import datetime, timedelta
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, TimeDistributed, Conv2D, ConvLSTM2D
import cartopy as crs
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cv2
from matplotlib.patches import Polygon



def copy_datasets():
    import shutil

    UPLOAD_DIR = "uploaded_dataset"
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    for i in range(len(dates_list)):
        if i % 20 == 0:
            for q in range (0, 6):
                year = dates_list[i+q].year
                if year < 2022:
                    continue
                os.makedirs(f'{UPLOAD_DIR}/dataset{i}', exist_ok=True)
                date = dates_list[i+q].date()
                from_file = f'/Users/muthumayan/Downloads/Images2/composite_data_S3_{date}.npy'
                to_file = f'./{UPLOAD_DIR}/dataset{i}/'
                shutil.copy(from_file, to_file)
                from_file = f'/Users/muthumayan/Downloads/Images2/composite_metadata_S3_{date}.pkl'
                to_file = f'./{UPLOAD_DIR}/dataset{i}/'
                shutil.copy(from_file, to_file)


# ### Globals


# Caution: Not a good idea to use globals in programming. But ....

# Contains listing of all wavelength in nm
G_Wavelengths = [0] * 1

# Max reflectance of each wavelength
GRmax = [0] * 1

# Reject images that have less than this many valid pixels.
# Meaning if there are too many NaNs, the image can skew training
G_validPixelsInImage = 5000

# Max value of chla to clamp to; Clamps outliers
G_maxChla = 600



# ## Utility Functions

# Files downloaded from earthaccess as granules. These are post-processed to aggregate 7 day moving average into a numpy file marked by a single date. It contains reading for all 22 channels for all lat, lon in one numpy array. The lat, lon information is in a separate metadate file.
# Note that though there are 22 channels, only 11 of them carry valid values. The rest of the channels are NaNs, except the last one channel-id 21 which carries the chla readings


def get_numpy_data_from_file(data_path=None, metadata_path=None):
    data = np.load(data_path)
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)

    # Bad coding habit, but using this for now
    global G_Wavelengths
    G_Wavelengths = metadata["wavelengths"][0:11]
    lat = metadata['lat']
    lon = metadata['lon']

    from datetime import datetime
    datestr = data_path.split('_')[-1]
    datestr = datestr.split('.')[0]
    start_date = datetime.strptime(datestr, "%Y-%m-%d")
    return (lat, lon, start_date, data)


# ### Plotting all wavelengths for a given day
# 
# This is meant to read the raw npy file and metadata and plot. The last channel (id: 21) is the 'chla' wavelength. It is not in the same scale as the other wavelengths. So, we'll exclude chla channel in this plot
def plot_day_spectrum_by_file(day_index, data_path="../Images/composite_data_S3_2019-01-07.npy", \
                      metadata_path="../Images/composite_metadata_S3_2019-01-07.pkl"):
    """
    Plots the mean spectrum for a given day index.

    Parameters:
        day_index (int): The index of the day (0 = first day in composite array).
        data_path (str): Path to the saved numpy array.
        metadata_path (str): Path to the saved metadata (pickle file).
    """
    # Load data and metadata
    data = np.load(data_path)
    print("data.shape:", data.shape)
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)

    #CHLA
    global G_Wavelengths
    G_Wavelengths = metadata["wavelengths"][0:11]
    print("G_Wavelengths:", G_Wavelengths)
    wavelengths = G_Wavelengths

    # Sanity check for day index
    if day_index < 0 or day_index >= data.shape[0]:
        print(f"Day index {day_index} is out of range. Data has {data.shape[0]} days.")
        return

    # Get the day's data
    day_data = data[day_index]  # shape (h, w, c)

    #CHLA
    day_data = day_data[:,:,0:11]

    # Average over h, w (ignoring NaNs)
    mean_spectrum = np.nanmean(day_data, axis=(0, 1))  # shape (c,)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(wavelengths, mean_spectrum, marker='o', linestyle='-')
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Mean Intensity (sr$^{-1}$ m$^{-2}$ nm$^{-1}$)")
    plt.title(f"Mean Spectrum for Day Index {day_index}")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


# ### Plotting all wavelengths for an image
# 
# The input is an image in the shape (93, 164, X). However, this function will just pick the first 11 channels. Plot the mean avergaes of the 11 different wavelengths for the entire lake image.

def plot_image_spectrum(image1, image2=None):

    """
    Plots the mean spectrum of image (93, 163, 14)

    Parameters:
        image (numpy array): The image (93, 163, 14)
    """

    global G_Wavelengths
    wavelengths = G_Wavelengths

    # Average over h, w (ignoring NaNs)
    mean_spectrum1 = np.nanmean(image1[:, :, :11], axis=(0, 1))  # shape (11,)
    if image2 is not None:
      mean_spectrum2 = np.nanmean(image2[:, :, :11], axis=(0, 1))  # shape (11,)

    print(mean_spectrum1.shape, wavelengths.shape)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(wavelengths, mean_spectrum1, marker='o', linestyle='-')
    if image2 is not None:
      plt.plot(wavelengths, mean_spectrum2, marker='o', linestyle='-.')
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Mean Intensity (sr$^{-1}$ m$^{-2}$ nm$^{-1}$)")
    plt.title(f"Mean Spectrum Image")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(['True', 'Pred'])
    plt.tight_layout()
    plt.show()


# ### Chlrophyll-a segmentation plot
# Since this reading has a large range from 0.1 to 1000, we we'll have to plot using log of the values. Chla readings are in band 21.
# 

def plot_chla_image(data_path="~/Downloads/Images2/composite_data_S3_2019-01-07.npy",
                    metadata_path="~/Downloads/Images2/composite_metadata_S3_2019-01-07.pkl",
                    band_index=21, min_thresh=2.5, max_thresh=7.0):

    lat, lon, start_date, data = get_numpy_data_from_file(data_path, metadata_path)

    # Plot band 10 (zero-based indexing)
    band_index = 21

    # need to use log because of the spread of data
    normd = np.log(data[0][:,:,band_index])

    # count of Nan and non-Nan data
    print(f'Nan count: {np.count_nonzero(np.isnan(normd))}')
    print(f'Non-Nan count: {np.count_nonzero(~np.isnan(normd))}')

    # Filter the higher value of chla
    mask = (normd > min_thresh) & (normd < max_thresh) & np.isfinite(normd)  # True between thresholds

    # Set elements where mask is False to NaN
    selected = np.where(mask, normd, np.nan)

    selected_after = selected.astype(np.uint8)

    # 4. Use OpenCV to find contours
    contours, _ = cv2.findContours(selected_after, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # --- Plot on Cartopy ---
    fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={'projection': ccrs.PlateCarree()})

    # Set map extent to Lake Erie region
    ax.set_extent([-83.62, -82.0, 41.34, 42.27], crs=ccrs.PlateCarree())
    #ax.set_extent([-83.25, -82.5, 41.4, 41.6], crs=ccrs.PlateCarree())

    # Add map features
    ax.coastlines(resolution='10m')
    ax.add_feature(cfeature.BORDERS)
    ax.add_feature(cfeature.LAKES, edgecolor='black', facecolor='none')
    ax.gridlines(draw_labels=True)

    # Plot reflectance
    im = ax.pcolormesh(lon, lat, normd, cmap='jet', shading='auto', vmin=0, vmax=7, transform=ccrs.PlateCarree())

    # Add colorbar and labels
    cbar = plt.colorbar(im, ax=ax, orientation='vertical', label='log(chloro_a)')


    # 5. Overlay contours back on map
    for cnt in contours:
        # cnt is an array of shape (N, 1, 2) with (row, col) indices
        cnt = cnt.squeeze()  # shape (N, 2)
        if cnt.ndim != 2:
            continue  # skip if contour is malformed

        # Convert from array indices to lat/lon
        y_idx, x_idx = cnt[:, 1], cnt[:, 0]
        contour_lons = lon[x_idx]
        contour_lats = lat[y_idx]

        #ax.plot(contour_lons, contour_lats, color='red', linewidth=1.2, transform=ccrs.PlateCarree())
        poly = Polygon(
            np.column_stack((contour_lons, contour_lats)),
            closed=True,
            facecolor='none',
            edgecolor='red',
            linewidth=0.8,
            alpha=1.0,
            transform=ccrs.PlateCarree()
        )
        ax.add_patch(poly)

    plt.title(f"Chlorophyll-a in range [{min_thresh}, {max_thresh}] around {start_date.date()}")
    #    plt.show()

    filename = './plotchla.png'
    
    # Save to a PNG file
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    return filename


def plot_chla_image_from_dataset(datasetdir, min_thresh=2.5, max_thresh=7.0):
    """
    Plots the chlorophyll-a image for a given dataset ID.
    
    Parameters:
        datasetid (int): The dataset ID to plot.
        min_thresh (float): Minimum threshold for chlorophyll-a.
        max_thresh (float): Maximum threshold for chlorophyll-a.
    """

    # Gather paths for data and metadata
    # and the files are named composite_data_S3_{date}.npy and composite_metadata_S3_{date}.pkl

    npy_files = [f for f in os.listdir(f'./{datasetdir}') if f.startswith('composite_data_S3_') and f.endswith('.npy')]
    pkl_files = [f for f in os.listdir(f'./{datasetdir}') if f.startswith('composite_metadata_S3_') and f.endswith('.pkl')]  

    # In-place sort files from oldest to newest
    npy_files.sort()
    pkl_files.sort()

    # Create a zip of the two lists
    files = list(zip(npy_files, pkl_files))

    if not npy_files or not pkl_files:
        raise FileNotFoundError(f"No data files found for dataset {datasetdir}. Please check the directory.")
    
    # Number of subplots (assumes 5 files)
    num_subplots = 5
    fig, axes = plt.subplots(nrows=1, ncols=num_subplots, figsize=(20, 4), 
                         subplot_kw={'projection': ccrs.PlateCarree()})

    # Ensure axes is always iterable
    if num_subplots == 1:
        axes = [axes]

    for i, f in enumerate(files[:num_subplots]):  # Limit to 5 files
        #print(f"Data file: {f[0]}, Metadata file: {f[1]}")
        data_path = f'./{datasetdir}/{f[0]}'
        metadata_path = f'./{datasetdir}/{f[1]}'
        lat, lon, start_date, data = get_numpy_data_from_file(data_path, metadata_path)

        band_index = 21
        normd = np.log(data[0][:, :, band_index])
        print(f'Normalized data shape: {normd.shape}')

        mask = (normd > min_thresh) & (normd < max_thresh) & np.isfinite(normd)
        
        ax = axes[i]
        ax.set_extent([-83.62, -82.0, 41.34, 42.27], crs=ccrs.PlateCarree())
        ax.coastlines(resolution='10m')
        ax.add_feature(cfeature.BORDERS)
        ax.add_feature(cfeature.LAKES, edgecolor='black', facecolor='none')
        ax.gridlines(draw_labels=True)  # Only show grid labels on first subplot
        ax.set_title(start_date.strftime('%Y-%m-%d'), fontsize=10)

        # Plot the data
        im = ax.pcolormesh(lon, lat, normd, cmap='jet', shading='auto', vmin=0, vmax=7, transform=ccrs.PlateCarree())

        #im = ax.imshow(normd, extent=[lon.min(), lon.max(), lat.min(), lat.max()],
        #            transform=ccrs.PlateCarree(), cmap='jet', vmin=0, vmax=7)

    # Adjust layout
    plt.tight_layout()
    filename = './plotchla.png'
    plt.savefig(filename, dpi=300)
    return filename


def plot_chla_true_pred_dataset(datasetdir, min_thresh=2.5, max_thresh=7.0):
    """
    Plots the chlorophyll-a image for a given dataset ID.
    
    Parameters:
        datasetid (int): The dataset ID to plot.
        min_thresh (float): Minimum threshold for chlorophyll-a.
        max_thresh (float): Maximum threshold for chlorophyll-a.
    """

    # Gather paths for data and metadata
    # and the files are named composite_data_S3_{date}.npy and composite_metadata_S3_{date}.pkl

    npy_files = [f for f in os.listdir(f'./{datasetdir}') if f.startswith('composite_data_S3_') and f.endswith('.npy')]
    pkl_files = [f for f in os.listdir(f'./{datasetdir}') if f.startswith('composite_metadata_S3_') and f.endswith('.pkl')]  

    lat, lon, start_date, data = get_numpy_data_from_file(f'./{datasetdir}/{npy_files[0]}', f'./{datasetdir}/{pkl_files[0]}')

    # In-place sort files from oldest to newest
    npy_files.sort()
    pkl_files.sort()

    # Create a zip of the two lists
    files = list(zip(npy_files, pkl_files))

    if not npy_files or not pkl_files:
        raise FileNotFoundError(f"No data files found for dataset {datasetdir}. Please check the directory.")
    
    # Number of subplots (assumes 2 )
    num_subplots = 2
    fig, axes = plt.subplots(nrows=1, ncols=num_subplots, figsize=(10, 4), 
                         subplot_kw={'projection': ccrs.PlateCarree()})

    # Ensure axes is always iterable
    if num_subplots == 1:
        axes = [axes]

    y_true, y_pred, y_date = inference_on_dataset(datasetdir)
    print(f'y_true: {y_true.shape}, y_pred: {y_pred.shape}, y_date: {y_date}')

    # this image from the training data or predicted data
    #Tchla = y_true[0][:,:,0]
    Tchla = y_true[0,:,:,0]
    print(f'Tchla shape: {Tchla.shape}')
    print(f'Image {y_date} Nan: {np.count_nonzero(np.isnan(Tchla))} Non-Nan: {np.count_nonzero(~np.isnan(Tchla))}')
    print(f'Image Min: {np.nanmin(Tchla)}, Max: {np.nanmax(Tchla)}')
    Tchla = np.log(Tchla)
    print(f'Image Min(log): {np.nanmin(Tchla)}, Max(log): {np.nanmax(Tchla)}')

    # this image from the training data or predicted data
    Tpred = y_pred[0,0,:,:,0]
    print(f'Image {y_date} Nan: {np.count_nonzero(np.isnan(Tpred))} Non-Nan: {np.count_nonzero(~np.isnan(Tpred))}')
    print(f'Image Min: {np.nanmin(Tpred)}, Max: {np.nanmax(Tpred)}')
    Tpred = np.log(Tpred)
    print(f'Image Min(log): {np.nanmin(Tpred)}, Max(log): {np.nanmax(Tpred)}')

    ax = axes[0]
    ax.set_extent([-83.62, -82.0, 41.34, 42.27], crs=ccrs.PlateCarree())
    ax.coastlines(resolution='10m')
    ax.add_feature(cfeature.BORDERS)
    ax.add_feature(cfeature.LAKES, edgecolor='black', facecolor='none')
    ax.gridlines(draw_labels=True)  # Only show grid labels on first subplot
    ax.set_title(f'Predicted: {y_date.date()}', fontsize=10)
    # Plot the data
    im = ax.pcolormesh(lon, lat, Tpred, cmap='jet', shading='auto', vmin=0, vmax=7, transform=ccrs.PlateCarree())

    ax = axes[1]
    ax.set_extent([-83.62, -82.0, 41.34, 42.27], crs=ccrs.PlateCarree())
    ax.coastlines(resolution='10m')
    ax.add_feature(cfeature.BORDERS)
    ax.add_feature(cfeature.LAKES, edgecolor='black', facecolor='none')
    ax.gridlines(draw_labels=True)  # Only show grid labels on first subplot
    ax.set_title(f'True: {y_date.date()}', fontsize=10)

    # Plot the data
    im = ax.pcolormesh(lon, lat, Tchla, cmap='jet', shading='auto', vmin=0, vmax=7, transform=ccrs.PlateCarree())

    # Adjust layout
    plt.tight_layout()
    filename = './plotchla.png'
    plt.savefig(filename, dpi=300)

    ssim_val = masked_ssim_np(y_true[0,:,:,0], y_pred[0,0,:,:,0], mask=None, max_val=G_maxChla)
    psnr = compute_psnr(y_true, y_pred, mask=None, max_val=G_maxChla)
    
    return filename, ssim_val, psnr


def plot_single_channel(lat, lon, start_date, data, waveindex, scaled=False):
    # Filter out specific wavelength
    rrs = data[0][:, :, waveindex]

    if scaled:
      vmin = -1
      vmax = 1
    else:
      vmin = np.nanmin(rrs)-0.01
      vmax = 1000.0

    print(f'Min: {np.nanmin(data)}, Max: {np.nanmax(data)}')

    # --- Plot on Cartopy ---
    fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={'projection': ccrs.PlateCarree()})

    # Set map extent to Lake Erie region
    ax.set_extent([-83.62, -82.0, 41.34, 42.27], crs=ccrs.PlateCarree())
    #ax.set_extent([-83.25, -82.5, 41.4, 41.6], crs=ccrs.PlateCarree())

    # Add map features
    ax.coastlines(resolution='10m')
    ax.add_feature(cfeature.BORDERS)
    ax.add_feature(cfeature.LAKES, edgecolor='black', facecolor='none')
    ax.gridlines(draw_labels=True)

    # Plot reflectance
    im = ax.pcolormesh(lon, lat, rrs, cmap='jet', shading='auto', vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())

    # Add colorbar and labels
    cbar = plt.colorbar(im, ax=ax, orientation='vertical', label=f'Wavelength index {waveindex}')

    plt.title(f"RRS [range: {vmin:.2f} to {vmax:.2f}] around {start_date.date()}")
    plt.show()


def normalize(data, eps=1e-8):
    # Incoming data (D, 93, 163, 1)
    D, H, W, C = data.shape
    normed_data = np.copy(data)
    stats = {}

    for c in range(C):
        values = data[..., c]
        median = np.nanmedian(values)
        q5 = np.nanpercentile(values, 10)
        q95 = np.nanpercentile(values, 90)
        iqr = q95 - q5 if q95 != q5 else 1.0

        normed_data[..., c] = (data[..., c] - median) / iqr
        stats[c] = (median, iqr)
        normed_data = np.tanh(normed_data)

    return normed_data, stats



def denormalize(normed_data, stats):
    """
    Reverse robust normalization for selected channels.

    Parameters:
        normed_data: normalized array (B, T, H, W, C)
        stats: dict from normalization step

    Returns:
        denormed_data: denormalized array
    """

    denormed_input = np.copy(normed_data)
    denormed_input = np.clip(denormed_input, -0.999999, 0.999999)
    denormed_scaled = np.arctanh(denormed_input)
    for c, (median, iqr) in stats.items():
        denormed_scaled[..., c] = denormed_scaled[..., c] * iqr + median
    return denormed_scaled

def loss_fn(y_true, y_pred):
    return masked_mse_loss(y_true, y_pred)

def masked_mse_loss(y_true, y_pred):
    print(f'MSE incoming: True: {y_true.shape}  Pred: {y_pred.shape}')
    # Split last channel as mask (assumes last channel is the mask)
    mask = y_true[..., -1]  # shape: (B, H, W)
    true = y_true[..., :1]  # shape: (B, H, W, C)
    print(mask.shape, true.shape, y_pred.shape)

    mask = tf.expand_dims(mask, axis=-1)  # shape: (B, H, W, 1)

    squared_error = tf.square(true - y_pred)
    masked_error = squared_error * mask

    # Avoid dividing by zero
    masked_mse = tf.reduce_sum(masked_error) / tf.reduce_sum(mask + 1e-8)
    return masked_mse


def prepare_dataset_for_inference(datasetdir):
    """
    Prepares the dataset for inference by concatenating the first 5 files in the dataset directory 
    and preparing them for model inference. The last file is used as the target for prediction.
    Parameters:
        datasetdir (str): Directory containing the dataset files.
    Returns:
        tuple: A tuple containing the prepared dataset and the target data.
    Raises:
        FileNotFoundError: If no data files are found in the specified directory.
    """
    global G_validPixelsInImage, G_maxChla, GRmax
    print(f"Preparing dataset for inference from directory: {datasetdir}")

    # Gather paths for data and metadata
    npy_files = [f for f in os.listdir(f'./{datasetdir}') if f.startswith('composite_data_S3_') and f.endswith('.npy')]
    pkl_files = [f for f in os.listdir(f'./{datasetdir}') if f.startswith('composite_metadata_S3_') and f.endswith('.pkl')]  

    # In-place sort files from oldest to newest
    npy_files.sort()
    pkl_files.sort()

    # Create a zip of the two lists
    allfiles = list(zip(npy_files, pkl_files))

    if not npy_files or not pkl_files:
        raise FileNotFoundError(f"No data files found for dataset {datasetdir}. Please check the directory.")
    
    # Initialize list to collect arrays
    array_list = []
    dates_list = []
    dropcount_allnan = 0
    dropcount_lesspixels = 0

    for npy,metanpy in list(allfiles):
        data_path = f'./{datasetdir}/{npy}'
        if not os.path.exists(data_path):
            print(f"Data file {data_path} does not exist. Skipping.")
            continue
        metadata_path = f'./{datasetdir}/{metanpy}'
        if not os.path.exists(metadata_path):
            print(f"Metadata file {metadata_path} does not exist. Skipping.")
            continue
        print(f"Data file: {data_path}, Metadata file: {metadata_path}")
        lat, lon, date, dailydata = get_numpy_data_from_file(data_path, metadata_path)


        if np.isnan(dailydata).all() == True:
            print(f'Skipping file {dailydata} as all data is NaN')
            dropcount_allnan += 1
            continue

        # count of non-Nan data for chla should at least be G_validPixelsInImage for good training
        if np.count_nonzero(~np.isnan(dailydata[:,:,:,21])) < G_validPixelsInImage:
            #print(f'Skipping file {dailydata} as less than {G_validPixelsInImage} non-NaN data')
            dropcount_lesspixels += 1
            continue

        array_list.append(dailydata)
        dates_list.append(date)

    # Stack them into shape (N, H, W, C)
    stacked_array = np.concatenate(array_list, axis=0)
    stacked_array = stacked_array[:,:,:,21:22] #ignore all other channels since they are NaNs
    stacked_array = np.clip(stacked_array, a_min = 0.001, a_max=G_maxChla)

    valid_mask = np.isfinite(stacked_array).astype(np.float32)  # 1 where finite, 0 where NaN
    scaled_array, GRmax = normalize(stacked_array, eps=1e-8)

    print(f'GRmax: {GRmax}')

    # Add two more dimensions for time
    scaled_array = add_time_features(scaled_array, dates_list)
    print(f"scaled array shape: {scaled_array.shape}, Dates count: {len(dates_list)}")

    # Create a mask channel across all data (across days, lat, lon, channels)
    mask = ~np.isnan(scaled_array[..., :1]).any(axis=-1, keepdims=True)
    print(f'Mask shape: {mask.shape}')
  
    # Replace NaNs with a -1.0 (lowest value)
    scaled_array[..., :1] = np.nan_to_num(scaled_array[..., :1], nan=-1.0)

    # Add the mask as a channel to the original array
    data = np.concatenate([scaled_array, mask], axis=-1)
    #data = scaled_array

    X = []
    y = []

    seq_len = 5  # Adjustable
    for i in range(len(data) - seq_len):
        print(f'X input: [{i, i+seq_len-1}], y input: {i+seq_len}, date: [{dates_list[i]}, {dates_list[i+seq_len-1]}]')
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])  # predict the next frame

    X = np.array(X)  # (samples, seq, H, W, channels)
    y = np.array(y)  # (samples, H, W, channels)

    X_test = X[0:1,...][:,:,:,:,0:3]
    y_test = y[0]

    print(X_test.shape, y_test.shape)

    return X_test, y_test, date, GRmax, valid_mask


def add_time_features(data, dates):
    T, H, W, C = data.shape
    #dates = [start_date + timedelta(days=8*i) for i in range(T)]
    day_of_year = np.array([d.timetuple().tm_yday for d in dates])
    sin_doy = np.sin(2 * np.pi * day_of_year / 365)
    cos_doy = np.cos(2 * np.pi * day_of_year / 365)

    sin_doy = sin_doy[:, None, None, None]
    cos_doy = cos_doy[:, None, None, None]

    sin_doy = np.tile(sin_doy, (1, H, W, 1))
    cos_doy = np.tile(cos_doy, (1, H, W, 1))

    return np.concatenate([data, sin_doy, cos_doy], axis=-1)  # (T, H, W, C+2)


def predict_and_denorm(model, X_test, y_test):
  global G_validPixelsInImage, G_maxChla, GRmax
  y_pred = model.predict(X_test)

  # The last channel in y_test is the mask channel
  valid_mask = y_test[..., -1].astype(bool)
  # Print the number of valid and invalid pixels
  print(f'Valid pixels: {np.count_nonzero(valid_mask)}, Invalid pixels: {np.count_nonzero(~valid_mask)}')
  true_frame = y_test[..., :1]

  #y_pred = np.expand_dims(y_pred, axis=0)  # Add batch dimension
  pred_frame = y_pred[..., :1]  # Only take the first channel (chla)

  # Apply mask to the predicted values
  print(f'Valid mask shape: {valid_mask.shape}, True frame shape: {true_frame.shape}, Pred frame shape: {pred_frame.shape}')
  true = np.where(valid_mask[..., np.newaxis], true_frame, np.nan)
  pred = np.where(valid_mask[..., np.newaxis], pred_frame, np.nan)

  # Add an extra dimension simulating the days dimension
  true = np.expand_dims(true, axis=0)
  pred = np.expand_dims(pred, axis=0)

  # Bring values back to [0,GRmax] range
  true = denormalize(true, GRmax)
  pred = denormalize(pred, GRmax)

  error = np.where(valid_mask[..., np.newaxis], pred - true, np.nan)
  return true, pred, error



def inference_on_dataset(datasetdir):
    """
    Performs inference on the dataset using the trained model.

    Parameters:
        model: The trained model.
        datasetdir (str): Directory containing the dataset files.

    Returns:
        tuple: A tuple containing the predicted data and the last date in the dataset.
    """
    global G_validPixelsInImage, G_maxChla, GRmax
    
    # Load from saved model
    from tensorflow.keras.models import load_model
    mymodel = load_model('/Users/muthumayan/Downloads/my_chla_model.keras', custom_objects={'loss_fn': loss_fn})

    X, y_true, y_date, GRmax, valid_mask = prepare_dataset_for_inference(datasetdir)
    
    # Predict the next frame
    true, pred, _ = predict_and_denorm(mymodel, X, y_true)

    
    return true, pred, y_date
   

def evaluate_metrics(y_true, y_pred, mask=None):
    # Assumes y_true and y_pred: (B, H, W, C)
    # mask: (B, H, W) or (B, H, W, 1)

    # Handle mask
    if mask is not None:
        if mask.ndim == 4:
            mask = mask[..., 0]
        mask = np.broadcast_to(mask[..., np.newaxis], y_true.shape)
    else:
        mask = np.isfinite(y_true)  # assume NaNs mask invalid pixels

    # MAE and RMSE per channel
    valid_count = np.sum(mask, axis=(0, 1, 2))  # per channel
    abs_error = np.abs(y_true - y_pred) * mask
    mse = ((y_true - y_pred) ** 2) * mask

    print(f'Masked bits: {np.isnan(mask).sum()}')

    mae_per_channel = np.nansum(abs_error, axis=(0, 1, 2)) / valid_count
    rmse_per_channel = np.sqrt(np.nansum(mse, axis=(0, 1, 2)) / valid_count)

    return mae_per_channel, rmse_per_channel

def compute_psnr(y_true, y_pred, mask=None, max_val=G_maxChla):
    # y_true, y_pred: shape (H, W, C) or (B, H, W, C)

    if mask is not None:
        mask = mask.astype(np.float32)
        mse = np.nansum(((y_true - y_pred) ** 2) * mask) / (np.sum(mask) + 1e-8)
    else:
        mse = np.nanmean((y_true - y_pred) ** 2)

    psnr = 10 * np.log10((max_val ** 2) / (mse + 1e-8))
    return psnr

import numpy as np
from skimage.metrics import structural_similarity as ssim
#from skimage.measure import compare_ssim as compare_ssim

def masked_ssim_np(y_true, y_pred, mask=None, max_val=G_maxChla):
    """
    Computes SSIM between y_true and y_pred, ignoring NaNs or using mask.
    y_true, y_pred: (H, W, C) or (H, W)
    mask: (H, W) or (H, W, C) boolean array (True = valid pixel)
    """
    if mask is None:
        mask = np.isfinite(y_true) & np.isfinite(y_pred)

    if y_true.ndim == 2:  # single-channel
        y_true = np.expand_dims(y_true, axis=-1)
        y_pred = np.expand_dims(y_pred, axis=-1)
        mask = np.expand_dims(mask, axis=-1)

    ssim_vals = []
    for c in range(y_true.shape[-1]):
        y_t = y_true[..., c]
        y_p = y_pred[..., c]
        m = mask[..., c]

        # Crop to only valid region
        y_t_valid = y_t[m]
        y_p_valid = y_p[m]

        if y_t_valid.size < 9:  # Too few valid pixels
            ssim_vals.append(np.nan)
            continue

        # Reshape to 2D square if possible
        size = int(np.sqrt(y_t_valid.size))
        if size * size < y_t_valid.size:
            size -= 1

        y_t_img = y_t_valid[:size*size].reshape((size, size))
        y_p_img = y_p_valid[:size*size].reshape((size, size))

        ssim_val = ssim(y_t_img, y_p_img, data_range=max_val, channel_axis=-1)
        #ssim_val = compare_ssim(y_t_img, y_p_img, data_range=max_val, multichannel=False)

        ssim_vals.append(ssim_val)

    return np.nanmean(ssim_vals)




