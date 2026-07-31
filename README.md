# Time-Series Final Project: Spatio-Temporal Dynamic Mapping

This repository contains the code for reconstructing and improving the methodology proposed in the paper:
**"Integrating Unsupervised Machine Intelligence and Anomaly Detection for Spatio-Temporal Dynamic Mapping Using Remote Sensing Image Series"** (Sustainability 2023).

## Project Overview

The goal of this project is to map landscape disturbances (such as the Mariana and Brumadinho dam collapses, and intense deforestation in Altamira) using unsupervised machine learning applied to time-series remote sensing data from Google Earth Engine (GEE).

### Methodology
1. **Paper Reconstruction (Stage 1):** We implemented the Isolation Forest (IF) and One-Class Support Vector Machine (OC-SVM) models to detect temporal anomalies.
2. **Simple Baseline:** We implemented a classical Z-Score thresholding model to serve as a naive statistical baseline.
3. **Improved Method (Stage 2):** We introduced a Deep Learning approach using an **LSTM Autoencoder**. Unlike the classical methods that evaluate isolated pixel values, the LSTM captures the sequential, temporal relationships of the landscape over time, calculating reconstruction errors to flag structural anomalies.

### Technical Specifications
- **Task Type:** Unsupervised Anomaly Detection (Multivariate input, Univariate output).
- **Sampling Frequency:** ~16 days (Terra MODIS), ~10-15 days (Landsat/Sentinel).
- **Input Window Length:** 3 time-steps (used for calculating Time-Aware Features like velocity, acceleration, and rolling stats for the LSTM).
- **Forecast Horizon:** N/A (Reconstruction-based detection, not future forecasting).
- **Random Seeds:** Fixed at `42` where possible to ensure reproducibility.

---

## Environment Requirements
Ensure you have Python 3.8+ installed. 

Install the required packages using the provided `requirements.txt`:
```bash
pip install -r requirements.txt
```

You must also authenticate your Google Earth Engine account before running the pipelines:
```bash
earthengine authenticate
```

---

## Execution Guide

### 1. Launch MLflow Dashboard
All experiments, parameters, and generated TIFFs are tracked automatically using MLflow. To start the local server (which includes a background daemon to prevent `desktop.ini` Google Drive syncing crashes):
```bash
python run_mlflow.py
```
*Access the dashboard at: http://localhost:5000*

### 2. Preprocess Data
Download and preprocess the GEE data (removes clouds/shadows, calculates spectral indices, and centers the median trend):
```bash
python run_preprocessing.py
```

### 3. Run Reconstructions & Baseline
Run the classical anomaly detection algorithms (IF, OC-SVM) for each dataset:
```bash
python Altamira_MODIS_repro.py
python Brumadinho_Sentinel_repro.py
python Mariana_Landsat_repro.py
```
Run the Simple Baseline (Z-Score) across all datasets:
```bash
python run_baseline_all.py
```

### 4. Run Improved Method (LSTM Autoencoder)
Train the deep learning model to learn the structural sequences of the environment:
```bash
python train_deep.py
```
Once trained, run the inference script to calculate reconstruction errors and map the anomalies:
```bash
python inference_deep.py
```

## Expected Outputs
- **MLflow Database:** All models will log their `total_anomalies`, `total_transitions`, and hyperparameters directly to the MLflow UI.
- **GeoTIFFs:** Georeferenced `.tif` anomaly maps will be saved in the `Tiff/` directory, categorized by algorithm and leak-free status.
