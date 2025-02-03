# Fault Detection in Wind Turbines Using Health Index Monitoring with Variational Autoencoders 

This repository contains our solution for the ASCE-EMI Structural Health Monitoring (SHM) for Wind Energy Challenge. We propose a Variational Autoencoder (VAE) based approach for fault detection in wind turbines. 

## Overview

Our solution addresses three critical fault detection tasks:
- Pitch drive failure detection
- Icing event detection
- Aerodynamic imbalance detection

The approach uses VAEs to learn a health index for each fault type, enabling robust and unsupervised fault detection in wind turbine systems.

## Project Structure

```
├── main/
│   ├── main_icing_detection.py          # Icing event detection
│   ├── main_imbalance_detection.py      # Aerodynamic imbalance detection
│   ├── main_pitch_fault_detection.py    # Pitch drive failure detection
│   └── main_wm_analysis.py              # SCADA data analysis and visualizing
├── src/
│   ├── model.py                         # Core model implementations
│   ├── utils.py                         # Utility functions
│   ├── utils_data_load.py               # Data loading utilities
│   └── utils_features.py                # Feature engineering utilities
├── dataset/                             # HDF5 files 
├── output_files_icing/                  # Output files for icing detection
├── output_files_imbalance/              # Output files for imbalance detection
├── output_files_pitch/                  # Output files for pitch fault detection
├── output_files_wm/                     # Output files for SCADA analysis
└── README.md                            # This file
```


## Requirements

- python 3.8+
- torch
- numPy
- pandas
- h5py
- scikit-learn
- matplotlib
- seaborn

Install all dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Data Preparation

1. Download the dataset from [Zenodo](https://doi.org/10.5281/zenodo.8229750)
2. Place all downloaded HDF5 files in the `dataset` directory

### Running Fault Detection Models

For each type of fault detection:

```bash
# Pitch Drive Fault Detection
python main/main_pitch_fault_detection.py

# Icing Detection
python main/main_icing_detection.py

# Imbalance Detection
python main/main_imbalance_detection.py
```


## Acknowledgments

- ASCE-EMI for organizing the challenge
- The challenge dataset was provided through Zenodo: 
  > Eleni Chatzi, Imad Abdallah, Martin Hofsäß, Oliver Bischoff, Sarah Barber, & Yuriy Marykovskiy. (2023). Aventa AV-7 ETH Zurich Research Wind Turbine SCADA and high frequency Structural Health Monitoring (SHM) data [Data set]. Zenodo.  https://doi.org/10.5281/zenodo.8229750 
