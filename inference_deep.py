import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time
import os
import mlflow
import feature_engineering as fe
import repro_utils
from models_deep import LSTMAutoencoder
from torch.utils.data import TensorDataset, DataLoader

def run_inference(dataset_name='Altamira', band_name='ndvi', batch_size=512):
    print(f"Loading {dataset_name} centered matrix for inference...")
    df = pd.read_parquet(f"Preprocess/{dataset_name}_{band_name.upper()}_CenteredMatrix.parquet")
    
    # Track valid pixels and dates
    dates = df.columns
    df_nan = df.replace(-99999, np.nan)
    
    print("Extracting Time-Aware Features...")
    feature_tensor = fe.build_time_aware_features(df_nan.values, window_size=3)
    feature_tensor = np.nan_to_num(feature_tensor, nan=0.0)
    
    x_tensor = torch.tensor(feature_tensor, dtype=torch.float32)
    dataset = TensorDataset(x_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- USING DEVICE: {device} ---")
    
    model = LSTMAutoencoder(num_features=5, hidden_dim=64, num_layers=2).to(device)
    
    # Load trained weights
    weights_path = "lstm_autoencoder.pth"
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Could not find trained weights at {weights_path}. Run train_deep.py first!")
    
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    
    criterion = nn.MSELoss(reduction='none') # Don't average, we need error per pixel
    
    print("Running Inference (Reconstructing sequences)...")
    all_errors = []
    
    with torch.no_grad():
        for batch_x in dataloader:
            batch_x = batch_x[0].to(device)
            reconstructed = model(batch_x)
            
            # Calculate MSE per pixel per timestep (average across the 5 features)
            loss = criterion(reconstructed, batch_x)
            pixel_time_error = loss.mean(dim=2).cpu().numpy()
            all_errors.append(pixel_time_error)
            
    # Concatenate all batches
    error_matrix = np.concatenate(all_errors, axis=0) # Shape: (Pixels, Time)
    
    # Restore NaNs where original data was missing
    original_nan_mask = np.isnan(df_nan.values)
    error_matrix[original_nan_mask] = np.nan
    
    # Thresholding: Calculate Mean + 3 * StdDev of the errors
    flat_errors = error_matrix[~np.isnan(error_matrix)]
    mean_err = np.mean(flat_errors)
    std_err = np.std(flat_errors)
    threshold = mean_err + 3 * std_err
    
    print(f"Error Threshold (Mean + 3*StdDev): {threshold:.6f}")
    
    # Flag anomalies
    # 1 for regular, -1 for anomaly
    pred_matrix = np.ones_like(error_matrix)
    pred_matrix[error_matrix > threshold] = -1
    pred_matrix[original_nan_mask] = np.nan
    
    # Create prediction DataFrame
    df_pred = pd.DataFrame(pred_matrix, index=df.index, columns=df.columns)
    
    # Calculate transitions/metrics using existing function
    df_metrics = repro_utils.calculate_metrics(df_pred)
    
    total_anomalies = int(df_metrics['Anomalias'].sum())
    total_transitions = int(df_metrics['Mudanças'].sum())
    print(f"Found {total_anomalies} total anomalies and {total_transitions} transitions!")
    
    # Log to MLflow
    mlflow.set_experiment(f"DynaLand_{dataset_name}_DeepLearning")
    with mlflow.start_run(run_name="LSTM_Autoencoder_Inference"):
        mlflow.log_param("threshold_strategy", "Mean + 3*StdDev")
        mlflow.log_param("threshold_value", threshold)
        mlflow.log_metric("total_anomalies", total_anomalies)
        mlflow.log_metric("total_transitions", total_transitions)
        
        # Save output TIFF
        tiff_dir = "Tiff/deep_learning/LSTM"
        os.makedirs(tiff_dir, exist_ok=True)
        tiff_filename = f"{dataset_name}_{band_name.upper()}_LSTM_Anomalies.tif"
        tiff_path = os.path.join(tiff_dir, tiff_filename)
        
        repro_utils.save_tiff_fromdf(df_metrics, ['Anomalias', 'p-valor'], -99999, tiff_path)
        mlflow.log_artifact(tiff_path)
        print(f"Successfully saved and logged deep learning map to {tiff_path}")

if __name__ == '__main__':
    run_inference()
