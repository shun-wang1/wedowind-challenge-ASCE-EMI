from src.utils_data_load import *
from src.utils import *
from src.model import *

def save_dataframe_to_csv(data, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for file_name, df in data.items():
        csv_path = os.path.join(output_dir, file_name.replace('.hdf5', '_SCADA.csv'))
        df.to_csv(csv_path, index=False)
        print(f"Features saved to {csv_path}")

def load_dataframe_from_csv(output_dir, hdf_files):
    files_to_read = [
        hdf_file.replace('.hdf5', '_SCADA.csv')
        for hdf_file in hdf_files
    ]

    data_frames = []
    for file_name in files_to_read:
        file_path = output_dir + file_name
        df = pd.read_csv(file_path)
        df['datetime'] = pd.to_datetime(df['file'].apply(extract_date)) + df['timestamp'].apply(extract_time_pd)
        data_frames.append(df)
    combined_data = pd.concat(data_frames, ignore_index=True)
    return combined_data

def plot_wm_channels(dataframe, output_dir):
    dataframe['date'] = dataframe['file'].apply(extract_date)
    unique_dates = dataframe['date'].dt.strftime('%Y-%m-%d').drop_duplicates().reset_index(drop=True)

    dataframe['Time_Index'] = range(len(dataframe))  # 创建时间索引作为横坐标

    channel_columns = [col for col in dataframe.columns if 'WM' in col or 'ATM' in col]

    #
    for channel in channel_columns:
        plt.figure(figsize=(12, 6))
        plt.plot(dataframe['Time_Index'], dataframe[channel], label=channel, linewidth=1.5)

        plt.xticks(
            ticks=dataframe['Time_Index'][dataframe['date'].dt.strftime('%Y-%m-%d').drop_duplicates().index],
            labels=unique_dates,
            rotation=45,
            ha='right'
        )

        plt.xlabel('Date')
        plt.ylabel('Signal Value')
        plt.title(f"Signal Trend for {channel}")
        plt.legend()
        plt.tight_layout()

        plt.savefig(f"{output_dir}/{channel}_trend.png")
        plt.close()

def plot_wind_speed_power_curve(df, start_date, end_date, wind_speed_col='WM2', power_col='WM3',
                                output_dir=None, save_name=None):
    df['date'] = pd.to_datetime(df['file'].str.extract(r'(\d{2}_\d{2}_\d{4})')[0], format='%d_%m_%Y')

    mask = (df['date'] >= pd.to_datetime(start_date)) & (df['date'] <= pd.to_datetime(end_date))
    filtered_df = df[mask]

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        filtered_df[wind_speed_col],
        filtered_df[power_col],
        color='#2878B5',
        s=15,
        alpha=0.6,
    )
    ax.set_xlim(0, 12)
    ax.set_ylim(-0.5, 7.5)
    ax.tick_params(axis='both', labelsize=16)
    ax.grid(True, linestyle='--', alpha=0.3, color='gray')
    ax.set_xlabel('Wind Speed (m/s)', fontsize=16)
    ax.set_ylabel('Power Output (MW)', fontsize=16)

    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

    time_range = f'Data Collection Period: {start_date} to {end_date}'
    fig.text(0.94, 0.14, time_range, fontsize=12, ha='right',
             style='italic', color='gray')

    plt.tight_layout()

    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Directory {output_dir} created.")

        save_path = os.path.join(output_dir, save_name)
        plt.savefig(save_path, bbox_inches='tight',dpi=600)  # 保存图像
        print(f"Saved plot to {save_path}")
    else:
        plt.show()

def preprocess_data(normal_data):
    df = normal_data.copy()

    df['datetime'] = pd.to_datetime(df['file'].apply(extract_date)) + df['timestamp'].apply(extract_time_pd)

    zero_power_datetimes = df[df['WM3'] <= 0]['datetime'].unique()
    out_of_speed_datetimes = df[(df['WM2'] < 2) | (df['WM2'] > 14)]['datetime'].unique()
    out_of_rotor_speed_datetimes = df[(df['WM1'] < 20) | (df['WM1'] > 66)]['datetime'].unique()

    invalid_datetimes = pd.unique(np.concatenate((zero_power_datetimes, out_of_speed_datetimes, out_of_rotor_speed_datetimes)))
    removed_datetimes = pd.DataFrame({
        'datetime': pd.to_datetime(np.concatenate((
            zero_power_datetimes,
            out_of_speed_datetimes,
            out_of_rotor_speed_datetimes
        ))),
        'reason': (['Zero Power'] * len(zero_power_datetimes) +
                   ['Out of Speed Range'] * len(out_of_speed_datetimes) +
                   ['Out of Rotor Speed Range'] * len(out_of_rotor_speed_datetimes))
    }).drop_duplicates()
    # df = df[~df['datetime'].isin(invalid_datetimes)]
    df = df[~df['datetime'].isin(zero_power_datetimes)]
    return df, removed_datetimes

def get_scada_filename(hdf_filename):
    # Aventa_Taggenberg_11_02_2022.hdf5" -> "Aventa_Taggenberg_11_02_2022_SCADA.csv"
    return hdf_filename.replace('.hdf5', '_SCADA.csv')

def main():
    set_seed(42)

    target_signals = [
        'WM1', 'WM2', 'WM3', 'WM4',
        'WM5',
        'ATM_TEMP_01',
        'ATM_HUM_01',
    ]
    base_path = r"../dataset"
    output_dir = r"../wm_files/"
    file_names = [
        "Aventa_Taggenberg_11_02_2022.hdf5",
        "Aventa_Taggenberg_14_02_2022.hdf5",
        "Aventa_Taggenberg_15_02_2022.hdf5",
        "Aventa_Taggenberg_16_02_2022.hdf5",

        "Aventa_Taggenberg_03_09_2022.hdf5",
        "Aventa_Taggenberg_01_11_2022.hdf5",
        "Aventa_Taggenberg_04_11_2022.hdf5",

        "Aventa_Taggenberg_17_12_2022.hdf5",
        "Aventa_Taggenberg_18_12_2022.hdf5",
        "Aventa_Taggenberg_19_12_2022.hdf5",
        "Aventa_Taggenberg_20_12_2022.hdf5",
        # #
        "Aventa_Taggenberg_08_12_2022.hdf5",
        "Aventa_Taggenberg_11_12_2022.hdf5",
        "Aventa_Taggenberg_19_12_2022.hdf5",
        "Aventa_Taggenberg_23_12_2022.hdf5",
        "Aventa_Taggenberg_29_12_2022.hdf5",
        "Aventa_Taggenberg_04_01_2023.hdf5",
        "Aventa_Taggenberg_15_01_2023.hdf5",
        "Aventa_Taggenberg_21_01_2023.hdf5",
    ]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Check if each SCADA file exists in output directory
    missing_scada = False
    for hdf_file in file_names:
        scada_file = get_scada_filename(hdf_file)
        if not os.path.exists(os.path.join(output_dir, scada_file)):
            print(f"Missing SCADA file in output directory: {scada_file}")
            missing_scada = True
            break
    if missing_scada:
        # Process HDF5 files if any SCADA file is missing
        print("Processing HDF5 files...")
        dataframe = process_hdf5_files_wm(base_path, file_names, target_signals)
        save_dataframe_to_csv(dataframe, output_dir)
    else:
        # Load existing files if all SCADA files are present
        print("Loading existing CSV files...")

    dataframe = load_dataframe_from_csv(output_dir, file_names)
    #####################################################
    df_processed, removed_datetimes = preprocess_data(dataframe)
    removed_datetimes.to_csv(output_dir+"removed_datetimes.csv", index=False)

    plot_wm_channels(df_processed, output_dir)
    plot_wind_speed_power_curve(
        df_processed,
        '2022-02-11',
        '2022-02-15',
        'WM2',
        'WM3',
        output_dir=output_dir,
        save_name="wind_speed_power_curve1.png"
    )
    plot_wind_speed_power_curve(
        df_processed,
        '2022-09-03',   #
        '2022-11-04',
        'WM2',
        'WM3',
        output_dir=output_dir,
        save_name="wind_speed_power_curve2.png"
    )

    df_processed_filtered = df_processed[['datetime', 'ATM_TEMP_01', 'ATM_HUM_01']]
    df_processed_filtered.to_csv(os.path.join(output_dir, 'ATM_data.csv'), index=False)
    print(f"Filtered data saved to {os.path.join(output_dir, 'ATM_data.csv')}")

if __name__ == "__main__":
    main()