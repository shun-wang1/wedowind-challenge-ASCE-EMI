from src.utils_features import *
from src.utils_data_load import *
from src.utils import *
from src.model import *
import time

def main():
    set_seed(42)

    target_signals = [
        'MSB_ACC_XX_01', 'MSB_ACC_ZZ_02',
        'MSH_ACC_XX_01', 'MSH_ACC_ZZ_01',
        'NMF_ACC_XX_02', 'NMF_ACC_YY_01', 'NMF_ACC_YY_02',
        'L5_ACC_XX_01', 'L5_ACC_YY_01',
        'L5_ACC_XX_02', 'L5_ACC_YY_02',
        'GEN_ACC_XX_01', 'GEN_ACC_YY_01', 'GEN_ACC_ZZ_01',
    ]
    base_path = r"../dataset"
    output_dir = r"../output_files_pitch/"
    file_names = [
        "Aventa_Taggenberg_11_02_2022.hdf5",
        "Aventa_Taggenberg_14_02_2022.hdf5",
        "Aventa_Taggenberg_15_02_2022.hdf5",
        "Aventa_Taggenberg_16_02_2022.hdf5",
    ]
    #
    cutoff_datetime = pd.to_datetime('2022-02-15 17:12:32')    # split for test data
    split_time = "2022-02-15 10:00:00"      # validation data 
    fault_time = '2022-02-16 17:12:32'      # stop due to pitch drive failure
    feature_file_names = convert_filenames(file_names)

    features_exist = all(
        os.path.exists(os.path.join(output_dir, feature_file))
        for feature_file in feature_file_names
    )
    if not features_exist:
        all_datasets, all_features, missing_value_info, error_values_info, wm_range_info = (
            process_hdf5_files(base_path, file_names, target_signals, cutoff_datetime))
        save_features_to_csv(all_features, output_dir)
        save_errors_to_json(error_values_info, "errors_output_pitch.json", output_dir=output_dir)
    else:
        print("Features already extracted, skipping processing step.")

    combined_data = load_data(output_dir, feature_file_names, cutoff_datetime)
    error_values_info = load_errors_from_json("errors_output_pitch.json", output_dir=output_dir)
    feature_columns = get_feature_columns(combined_data, target_signals)

    (train_dataset, val_dataset, icing_dataset,
     fault_record_index, all_record_index, dates_for_legend, all_datetimes) = (
        split_data(combined_data, feature_columns, fault_time, split_time, error_values_info, output_dir))

    input_dim = train_dataset.input_dim
    print(input_dim)
    model = AutoEncoderVAE(input_dim)

    batch_size = 256
    train_loader, val_loader, icing_test_loader = create_data_loaders(
        train_dataset, val_dataset, icing_dataset, batch_size
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #
    start_time = time.time()
    train_model(model, train_loader, num_epochs=100, learning_rate=1e-3, device=device)
    end_time = time.time()
    training_time = end_time - start_time
    print(f"Training completed in {training_time:.2f} seconds")

    train_re_scores = calculate_re(model, train_loader, device=device)
    val_re_scores = calculate_re(model, val_loader, device=device)
    test_re_scores = calculate_re(model, icing_test_loader, device=device)

    train_HI, val_HI, test_HI, threshold = process_health_indices(train_re_scores, val_re_scores, test_re_scores)

    plot_hi(train_HI, val_HI, test_HI, threshold, fault_record_index, all_record_index,
            dates_for_legend, all_datetimes, output_dir)


if __name__ == "__main__":
    main()