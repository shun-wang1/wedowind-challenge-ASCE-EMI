from src.utils_features import *
from src.utils_data_load import *
from src.utils import *
from src.model import *
import time

def main():
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    target_signals = [
        'MSB_ACC_XX_01', 'MSB_ACC_ZZ_02',
        'MSH_ACC_XX_01', 'MSH_ACC_ZZ_01',
        'NMF_ACC_XX_02', 'NMF_ACC_YY_01', 'NMF_ACC_YY_02',
        'L5_ACC_XX_01', 'L5_ACC_YY_01',
        'L5_ACC_XX_02', 'L5_ACC_YY_02',
        'GEN_ACC_XX_01', 'GEN_ACC_YY_01', 'GEN_ACC_ZZ_01',
    ]
    base_path = r"../dataset"
    output_dir = r"../output_files_imbalance/"
    file_names = [
    "Aventa_Taggenberg_03_09_2022.hdf5",
    "Aventa_Taggenberg_01_11_2022.hdf5",
    "Aventa_Taggenberg_04_11_2022.hdf5",
    #
    "Aventa_Taggenberg_08_12_2022.hdf5",
    "Aventa_Taggenberg_11_12_2022.hdf5",
    "Aventa_Taggenberg_19_12_2022.hdf5",
    "Aventa_Taggenberg_23_12_2022.hdf5",
    "Aventa_Taggenberg_29_12_2022.hdf5",
    "Aventa_Taggenberg_04_01_2023.hdf5",
    "Aventa_Taggenberg_15_01_2023.hdf5",
    "Aventa_Taggenberg_21_01_2023.hdf5",
    ]

    cutoff_datetime = pd.to_datetime('2022-11-04 00:00:00')   # split for test data
    split_time = '2022-11-01 14:00:00'      # validation data 
    fault_time = '2022-12-08 00:03:09'  # imbalance fault
    feature_file_names = convert_filenames(file_names)

    features_exist = all(
        os.path.exists(os.path.join(output_dir, feature_file))
        for feature_file in feature_file_names
    )
    if not features_exist:
        all_datasets, all_features, missing_value_info, error_values_info, wm_range_info = (
            process_hdf5_files(base_path, file_names, target_signals, cutoff_datetime))
        save_features_to_csv(all_features, output_dir)
        save_errors_to_json(error_values_info, "errors_output_icing.json", output_dir=output_dir)
    else:
        print("Features already extracted, skipping processing step.")

    combined_data = load_data(output_dir, feature_file_names, cutoff_datetime)
    error_values_info = load_errors_from_json("errors_output_icing.json",output_dir=output_dir)
    feature_columns = get_feature_columns(combined_data, target_signals)

    (train_dataset, val_dataset, test_dataset,
     fault_record_index, all_record_index, dates_for_legend, all_datetimes) = (
        split_data(combined_data, feature_columns, fault_time, split_time, error_values_info,output_dir))

    input_dim = train_dataset.input_dim
    model = AutoEncoderVAE(input_dim)

    batch_size = 256
    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset, val_dataset, test_dataset, batch_size
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 训练自编码器
    start_time = time.time()
    train_model(model, train_loader, num_epochs=100, learning_rate=1e-3, device=device)
    end_time = time.time()
    training_time = end_time - start_time
    print(f"Training completed in {training_time:.2f} seconds")

    #  HI
    train_re_scores = calculate_re(model, train_loader, device=device)
    val_re_scores = calculate_re(model, val_loader, device=device)
    test_re_scores = calculate_re(model, test_loader, device=device)

    train_HI, val_HI, test_HI, threshold = process_health_indices(train_re_scores, val_re_scores, test_re_scores)

    plot_hi(train_HI, val_HI, test_HI, threshold, fault_record_index, all_record_index,
            dates_for_legend, all_datetimes, output_dir)

    all_hi = np.concatenate([train_HI, val_HI, test_HI])
    test_start_index = len(train_HI) + len(val_HI)
    test_hi_scores = all_hi[test_start_index:]  # Extract test data portion from all health index scores
    fault_index_in_test = fault_record_index - test_start_index  # Calculate fault index relative to test data start

    # Split test data into normal and faulty periods
    normal_test_hi_scores = test_hi_scores[:fault_index_in_test]
    fault_test_hi_scores = test_hi_scores[fault_index_in_test:]

    predictions, true_labels, metrics = evaluate_test_performance(
        normal_test_hi_scores, fault_test_hi_scores, threshold, output_dir
    )
    # Save prediction results and ground truth labels
    save_predictions_and_labels(predictions, true_labels, "predictions_and_labels_imbalance.csv", output_dir=output_dir)


if __name__ == "__main__":
    main()