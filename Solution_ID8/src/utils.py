import torch
from typing import List
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
import re
import pandas as pd
from datetime import datetime, time, timedelta
from torch.utils.data import Dataset, DataLoader
import json
import pickle
import random
from pathlib import Path

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_feature_columns(dataframe, feature_keywords):
    pattern = re.compile('|'.join([re.escape(keyword) for keyword in feature_keywords]), re.IGNORECASE)
    feature_columns = [col for col in dataframe.columns if pattern.search(col)]
    return feature_columns

def extract_date(file_name):
    date_str = '_'.join(file_name.split('_')[-3:]).replace('.hdf5', '')
    return datetime.strptime(date_str, "%d_%m_%Y")

def extract_time_pd(time_str):
    hours, minutes, seconds = map(int, time_str.split('_'))
    return pd.Timedelta(hours=hours, minutes=minutes, seconds=seconds)

def load_data(directory_path, files_to_read, cutoff_datetime):
    data_frames = []
    for file_name in files_to_read:
        file_path = directory_path + file_name
        df = pd.read_csv(file_path)
        df['datetime'] = pd.to_datetime(df['file'].apply(extract_date)) + df['timestamp'].apply(extract_time_pd)
        df['Usage'] = df['datetime'].apply(lambda dt: 'Normal' if dt <= cutoff_datetime else 'Testing')
        data_frames.append(df)
    combined_data = pd.concat(data_frames, ignore_index=True)
    return combined_data

def load_errors_from_json(file_path, output_dir=None):
    if output_dir:
        file_path = os.path.join(output_dir, file_path)
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)
    results = []
    for file_name, times in data.items():
        parts = file_name.split('_')
        date_part = parts[2] + "-" + parts[3] + "-" + parts[4].split('.')[0]
        for time in times:
            hour, minute, second = time.split('_')
            time_formatted = f"{hour}:{minute}:{second}"
            full_datetime = pd.to_datetime(f"{date_part} {time_formatted}", format="%d-%m-%Y %H:%M:%S")
            results.append((full_datetime))

    df = pd.DataFrame(results, columns=["datetime"])
    return df


def convert_filenames(file_names: List[str]) -> List[str]:
    """Convert HDF5 filenames to corresponding feature CSV filenames"""
    return [f"{Path(name).stem}_features.csv" for name in file_names]

def extract_date_time(dataframe):
    df = dataframe.copy()
    df['datetime'] = pd.to_datetime(df['file'].apply(extract_date)) + df['timestamp'].apply(extract_time_pd)
    return df

def filter_zero_power_rows(df, removed_datetimes_path):
    removed_datetimes = pd.read_csv(removed_datetimes_path)
    invalid_datetimes = removed_datetimes[
        (removed_datetimes['reason'] == 'Zero Power')
            #  | (removed_datetimes['reason'] == 'Out of Speed Range')
            # | (removed_datetimes['reason'] == 'Out of Rotor Speed Range')
        ]['datetime']

    invalid_datetimes  = pd.to_datetime(invalid_datetimes)
    cleaned_df = df[~df['datetime'].isin(invalid_datetimes )]
    return cleaned_df

def split_data(combined_data, feature_columns, fault_time, split_datetime, errors_df, output_dir):
    # Separate normal and test data
    normal_features_with_aux = combined_data[combined_data['Usage'] == 'Normal'].copy()
    test_features_with_aux = combined_data[combined_data['Usage'] == 'Testing'].copy()
    normal_features_with_aux= extract_date_time(normal_features_with_aux)
    test_features_with_aux = extract_date_time(test_features_with_aux)
    errors_df['datetime'] = pd.to_datetime(errors_df['datetime'])

    # Remove error records from normal data
    normal_features_with_aux = normal_features_with_aux.merge(errors_df, on='datetime', how='left', indicator=True)
    normal_features_with_aux = normal_features_with_aux[normal_features_with_aux['_merge'] == 'left_only'].drop(columns=['_merge'])
    # normal_features_with_aux = filter_zero_power_rows(normal_features_with_aux, 'wm_files/removed_datetimes.csv')

    test_features_with_aux = test_features_with_aux.merge(errors_df, on='datetime', how='left', indicator=True)
    test_features_with_aux = test_features_with_aux[test_features_with_aux['_merge'] == 'left_only'].drop(columns=['_merge'])
    # test_features_with_aux = filter_zero_power_rows(test_features_with_aux, 'wm_files/removed_datetimes.csv')

    split_datetime = pd.to_datetime(split_datetime)
    train_datasets = normal_features_with_aux[normal_features_with_aux['datetime'] <= split_datetime]
    val_datasets = normal_features_with_aux[normal_features_with_aux['datetime'] > split_datetime]
    test_datasets = test_features_with_aux[:]

    sample_counts_train = record_count(train_datasets)
    sample_counts_val = record_count(val_datasets)
    sample_counts_test = record_count(test_datasets)

    print("Train Data Sample Counts by Date:")
    for date, count in sample_counts_train.items():
        print(f"{date}: {count:.2f}")

    print("Validation Data Sample Counts by Date:")
    for date, count in sample_counts_val.items():
        print(f"{date}: {count:.2f}")

    print("Test Data Sample Counts by Date:")
    for date, count in sample_counts_test.items():
        print(f"{date}: {count:.2f}")

    combined_data = pd.concat([train_datasets, val_datasets, test_datasets], ignore_index=True)

    fault_record_index = find_matching_indices(combined_data, fault_time)
    all_record_index, dates_for_legend = find_first_timestamps(combined_data)

    train_features = train_datasets[feature_columns]
    val_features = val_datasets[feature_columns]
    test_features = test_datasets[feature_columns]

    # Normalization
    scaler = MinMaxScaler()  # StandardScaler()   MinMaxScaler()
    train_samples = scaler.fit_transform(train_features)
    val_samples = scaler.transform(val_features)
    test_samples = scaler.transform(test_features)

    scaler_path = os.path.join(output_dir, "scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    train_dataset = CustomDataset(train_samples)
    val_dataset = CustomDataset(val_samples)
    test_dataset = CustomDataset(test_samples)
    all_datetimes = pd.concat([train_datasets['datetime'], val_datasets['datetime'], test_datasets['datetime']])
    all_datetimes = all_datetimes.drop_duplicates().reset_index(drop=True)
    #
    return (train_dataset, val_dataset, test_dataset,
            fault_record_index, all_record_index, dates_for_legend, all_datetimes)

def create_data_loaders(train_dataset, val_dataset, test_dataset, batch_size=32):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

class CustomDataset(Dataset):
    def __init__(self, features):
        self.features = torch.tensor(features).float()
        # Store the feature dimension
        self.feature_dim = self.features.shape[1] if self.features.ndim > 1 else 1

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        data = self.features[idx]
        return data

    @property
    def input_dim(self):
        """Return the dimension of the input features."""
        return self.feature_dim

def record_count(df):
    dates = pd.to_datetime(df['file'].apply(extract_date))  # 确保是 datetime 对象
    sample_counts = dates.value_counts().sort_index()
    record_counts = sample_counts / 60
    return  record_counts

def extract_time_str(time_str):
    h, m, s = map(int, time_str.split('_'))
    return time(h, m, s)

def find_matching_indices(df, datetime_str):
    extracted_dates = pd.to_datetime(df['file'].apply(extract_date))
    extracted_times = df['timestamp'].apply(extract_time_str)
    specific_datetime = pd.to_datetime(datetime_str)
    matching_indices = df[(extracted_dates.dt.date == specific_datetime.date()) &
                          (extracted_times == specific_datetime.time())].index
    if len(matching_indices) != 60:
        raise ValueError(f"Expected 60 matching indices, but found {len(matching_indices)}.")
    return min(matching_indices) // 60

def find_first_timestamps(df):
    extracted_dates = pd.to_datetime(df['file'].apply(extract_date))
    extracted_times = df['timestamp'].apply(extract_time_pd)

    complete_datetimes = extracted_dates + extracted_times
    first_timestamp_indices = complete_datetimes.groupby(complete_datetimes.dt.date).idxmin()
    record_indices = first_timestamp_indices // 60
    dates_for_legend = complete_datetimes.loc[first_timestamp_indices].dt.strftime('%Y-%m-%d').tolist()

    return record_indices, dates_for_legend

