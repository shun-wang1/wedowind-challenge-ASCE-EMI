import h5py
import os
from datetime import datetime
import numpy as np
import pickle
import pandas as pd
from scipy.signal import detrend
from src.utils_features import *
import scipy.stats as stats
import sys
from scipy.interpolate import interp1d
import json

def extract_date(file_name):
    date_str = '_'.join(file_name.split('_')[-3:]).replace('.hdf5', '')
    return datetime.strptime(date_str, "%d_%m_%Y")

def extract_time_pd(time_str):
    hours, minutes, seconds = map(int, time_str.split('_'))
    return pd.Timedelta(hours=hours, minutes=minutes, seconds=seconds)


def should_skip_timestamp(f, dataset_name, time_stamp):
    target_signal = 'MSH_ACC_XX_01'  # 直接指定需要检查的列名
    signal_length = len(f[dataset_name][time_stamp][target_signal]['Value'][()])
    return signal_length < 120000

def process_hdf5_files(base_path, file_names, target_signals, cutoff_date):
    wm_signals = ['WM1', 'WM2', 'WM3', 'WM4', 'WM5']
    sorted_file_names = sorted(file_names, key=extract_date)
    all_datasets = {}
    all_features = {}
    error_values_info = {}
    missing_value_info = {}
    wm_range_info = {}

    # Process each file
    for file_name in sorted_file_names:
        file_path = os.path.join(base_path, file_name)
        file_date = extract_date(file_name)
        datasets = {}
        features_data = []

        missing_values = []
        error_values = []
        wm_ranges = []
        # Open and read HDF5 file
        with h5py.File(file_path, "r") as f:
            print(f"Processing file: {file_path}")
            for dataset_name in f.keys():
                datasets[dataset_name] = {}
                for time_stamp in f[dataset_name].keys():
                    signal_list = list(f[dataset_name][time_stamp].keys())

                    # Remove 'ChannelList' if it exists
                    if 'ChannelList' in signal_list:
                        signal_list.remove('ChannelList')

                    # Skip this entire time_stamp if any target signal is missing
                    if not all(signal in f[dataset_name][time_stamp] for signal in target_signals):
                        print(f"Skipping timestamp {time_stamp} - target signal is missing")
                        continue

                    # Check if data length is less than 10 minutes (120000 points)
                    if should_skip_timestamp(f, dataset_name, time_stamp):
                        continue

                    segments_data = []
                    print(time_stamp)

                    # Additional range checks for WM signals
                    for wm in wm_signals:
                        if wm in signal_list:
                            wm_values = f[dataset_name][time_stamp][wm]['Value'][()]
                            wm_min, wm_max = np.min(wm_values), np.max(wm_values)
                            wm_ranges.append((time_stamp, wm, wm_min, wm_max))

                    wm1_check = False
                    wm2_check = False
                    wm3_check = False
                    date_time = pd.to_datetime(extract_date(file_name)) + extract_time_pd(time_stamp)
                    if date_time <= cutoff_date:
                        wm3_check = not np.all(
                            f[dataset_name][time_stamp]['WM3']['Value'][()] > 0
                        )
                    if wm1_check or wm2_check or wm3_check:
                        error_values.append(time_stamp)
                        continue

                    datasets[dataset_name][time_stamp] = {}
                    for signal in target_signals:
                        datasets[dataset_name][time_stamp][signal] = {}
                        time_data = f[dataset_name][time_stamp][signal]['Time'][()]
                        value_data = f[dataset_name][time_stamp][signal]['Value'][()]

                        # Check for missing values (NaNs)
                        if np.any(np.isnan(time_data)):
                            missing_values.append(f"Missing 'Time' values in {dataset_name}/{time_stamp}/{signal}")
                        if np.any(np.isnan(value_data)):
                            missing_values.append(f"Missing 'Value' values in {dataset_name}/{time_stamp}/{signal}")

                        # Split value_data into smaller segments (e.g., 1-second segments)
                        sample_length = 2000  # 200 samples per second for a 200Hz frequency
                        sampling_frequency = 200  # 200Hz
                        value_data = np.squeeze(value_data)

                        if signal in wm_signals:
                            original_indices = np.linspace(0, len(value_data) - 1, num=len(value_data))
                            new_indices = np.linspace(0, len(value_data) - 1, 60)

                            if signal == 'WM1':
                                value_data = pd.Series(value_data).replace(0, np.nan)
                                value_data = value_data.fillna(method='ffill').fillna(method='bfill').to_numpy()
                            interpolator = interp1d(original_indices, value_data, kind='linear', fill_value='extrapolate')
                            resampled_values = interpolator(new_indices)
                            for segment_index in range(60):
                                if len(segments_data) <= segment_index:
                                    segments_data.append({
                                        'file': file_name,
                                        'timestamp': time_stamp,
                                        'segment_index': segment_index
                                    })
                                segments_data[segment_index][f'{signal}'] = resampled_values[segment_index]

                        elif signal in ['ATM_TEMP_01', 'ATM_HUM_01']:
                            filled_data = fill_outliers(value_data, lower_limit=-200, upper_limit=200)
                            segment_length = len(filled_data) // 60
                            averaged_values = [np.mean(filled_data[i * segment_length:(i + 1) * segment_length]) for i in
                                               range(60)]
                            for segment_index in range(60):
                                if len(segments_data) <= segment_index:
                                    segments_data.append({
                                        'file': file_name,
                                        'timestamp': time_stamp,
                                        'segment_index': segment_index
                                    })
                                segments_data[segment_index][f'{signal}'] = averaged_values[segment_index]

                        else:
                            value_data_samples, time_data_samples = split_data_with_time(value_data, sample_length, sampling_frequency)
                            features_per_segment = calculate_features_per_segment(time_data_samples, value_data_samples)
                            for segment_index, feature_dict in enumerate(features_per_segment):
                                if len(segments_data) <= segment_index:
                                    segments_data.append({
                                        'file': file_name,
                                        'timestamp': time_stamp,
                                        'segment_index': segment_index
                                    })
                                for feature_name, feature_value in feature_dict.items():
                                    segments_data[segment_index][f'{signal}_{feature_name}'] = feature_value

                            # Store time and value data
                            datasets[dataset_name][time_stamp][signal]['Time'] = time_data_samples
                            datasets[dataset_name][time_stamp][signal]['Value'] = value_data_samples

                    features_data.extend(segments_data)

        # Record missing or error values information, if any
        if missing_values:
            missing_value_info[file_name] = missing_values
        if error_values:
            error_values_info[file_name] = error_values
        if wm_ranges:
            wm_range_info[file_name] = wm_ranges

        all_datasets[file_name] = datasets
        df_features = pd.DataFrame(features_data)
        all_features[file_name] = df_features
    return all_datasets, all_features, missing_value_info, error_values_info, wm_range_info

def fill_outliers(value_data, lower_limit=0, upper_limit=100):
    value_data = pd.Series(value_data)
    value_data[(value_data < lower_limit) | (value_data > upper_limit)] = np.nan
    value_data = value_data.fillna(method='ffill').fillna(method='bfill')
    return value_data.to_numpy()

def calculate_features_per_segment(time_data_samples, value_data_samples):
    features_list = []
    for time_data, value_data in zip(time_data_samples, value_data_samples):
        if value_data.ndim > 1:
            value_data = value_data.flatten()
        value_data = detrend(value_data, type='constant')   # value_data = value_data - np.mean(value_data)
        features = {
            # Time domain features
            'StdDev': np.std(value_data),
            'RMS': rms_fea(value_data),
            'PeakToPeak': pp_fea(value_data),
            'Skewness': stats.skew(value_data),
            'Kurtosis': stats.kurtosis(value_data),
            'Absolute Max': max_fea(value_data),
            'Absolute Mean': np.mean(np.abs(value_data)),
            'Min': np.min(value_data),
            'Shape Factor': shape_factor(value_data),
            'Peak Factor': peak_factor(value_data),
            'Impulse Factor': impluse_factor(value_data),
            'Crest Factor': crest_factor(value_data),
            'Clearance Factor': clearance_factor(value_data),
            'Spectral Entropy': fft_entropy(value_data),
            'Spectral Mean': fft_mean(value_data),
            'Spectral Var': fft_var(value_data),
            'Spectral Std': fft_std(value_data),
            'Spectral Energy': fft_energy(value_data),
            'Spectral Skewness': fft_skew(value_data),
            'Spectral Kurtosis': fft_kurt(value_data),
        }
        features_list.append(features)
    return features_list

def split_data_with_time(value_data, sample_length, sampling_frequency):
    num_samples = len(value_data)
    time_interval = 1 / sampling_frequency
    segments = []
    time_segments = []

    for start in range(0, num_samples, sample_length):
        end = start + sample_length
        if end > num_samples:
            break

        # Split value_data and generate corresponding time_data
        segments.append(value_data[start:end])
        time_segments.append(np.arange(start * time_interval, end * time_interval, time_interval))

    return segments, time_segments



def save_features_to_csv(data, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file_name, df in data.items():
        csv_path = os.path.join(output_dir, file_name.replace('.hdf5', '_features.csv'))
        df.to_csv(csv_path, index=False)
        print(f"Features saved to {csv_path}")

def print_missing_value_info(missing_value_info):
    if missing_value_info:
        print("Missing values found in the following files:")
        for file, missing_info in missing_value_info.items():
            print(f"\nFile: {file}")
            for info in missing_info:
                print(f" - {info}")
    else:
        print("No missing values found.")

def print_wm_range_info(wm_range_info):
    if wm_range_info:
        print("WM range information:")
        for file, ranges in wm_range_info.items():
            print(f"\nFile: {file}")
            for (timestamp, wm, wm_min, wm_max) in ranges:
                print(f" - {timestamp}: {wm} min = {wm_min}, max = {wm_max}")
    else:
        print("No WM range information found.")


def save_errors_to_json(error_values_info, output_file, output_dir=None):
    if output_dir:
        output_file = os.path.join(output_dir, output_file)
    with open(output_file, 'w') as json_file:
        json.dump(error_values_info, json_file, indent=4)


def process_hdf5_files_wm(base_path, file_names, target_wm_signals):
    sorted_file_names = sorted(file_names, key=extract_date)
    all_features = {}
    wm_signals = ['WM1', 'WM2', 'WM3', 'WM4', 'WM5']

    for file_name in sorted_file_names:
        file_path = os.path.join(base_path, file_name)
        datasets = {}
        features_data = []
        with h5py.File(file_path, "r") as f:
            print(f"Processing file: {file_path}")
            for dataset_name in f.keys():
                datasets[dataset_name] = {}
                for time_stamp in f[dataset_name].keys():
                    signal_list = list(f[dataset_name][time_stamp].keys())
                    segments_data = []

                    # Remove 'ChannelList' if it exists
                    if 'ChannelList' in signal_list:
                        signal_list.remove('ChannelList')

                    # Only process WM signals
                    wm_signals_present = [wm for wm in target_wm_signals if wm in signal_list]
                    datasets[dataset_name][time_stamp] = {}

                    for signal in wm_signals_present:
                        value_data = f[dataset_name][time_stamp][signal]['Value'][()]
                        value_data = np.squeeze(value_data)
                        if signal in wm_signals:
                            original_indices = np.linspace(0, len(value_data) - 1, num=len(value_data))
                            new_indices = np.linspace(0, len(value_data) - 1, 60)

                            if signal == 'WM1':
                                value_data = pd.Series(value_data).replace(0, np.nan)
                                nan_indices = value_data[value_data.isna()].index
                                window = 10
                                for idx in nan_indices:
                                    start_idx = max(0, idx - window // 2)
                                    end_idx = min(len(value_data), idx + window // 2 + 1)
                                    local_mean = value_data[start_idx:end_idx].mean(skipna=True)
                                    value_data[idx] = local_mean  # 用局部平均填充 NaN
                                value_data = value_data.fillna(method='ffill').fillna(method='bfill').to_numpy()

                            interpolator = interp1d(original_indices, value_data, kind='linear',
                                                    fill_value='extrapolate')
                            resampled_values = interpolator(new_indices)
                            for segment_index in range(60):
                                if len(segments_data) <= segment_index:
                                    segments_data.append({
                                        'file': file_name,
                                        'timestamp': time_stamp,
                                        'segment_index': segment_index
                                    })
                                segments_data[segment_index][f'{signal}'] = resampled_values[segment_index]

                        elif signal in ['ATM_TEMP_01', 'ATM_HUM_01']:
                            filled_data = fill_outliers(value_data, lower_limit=-200, upper_limit=200)
                            segment_length = len(filled_data) // 60
                            averaged_values = [np.mean(filled_data[i * segment_length:(i + 1) * segment_length]) for i in
                                               range(60)]

                            for segment_index in range(60):
                                if len(segments_data) <= segment_index:
                                    segments_data.append({
                                        'file': file_name,
                                        'timestamp': time_stamp,
                                        'segment_index': segment_index
                                    })
                                segments_data[segment_index][f'{signal}'] = averaged_values[segment_index]


                        datasets[dataset_name][time_stamp][signal] = value_data
                    features_data.extend(segments_data)
        df_features = pd.DataFrame(features_data)
        all_features[file_name] = df_features
    return all_features

