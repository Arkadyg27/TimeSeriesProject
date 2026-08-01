import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
import repro_utils
import feature_engineering as fe
from models_deep import LSTMAutoencoder

def compute_custom_unsupervised_metrics(df_pred):
    """
    Computes custom unsupervised metrics for temporal-spatial time series anomaly detection:
    1. Temporal Persistence Ratio (TPR): Fraction of anomalies that persist for >= 2 consecutive dates
    2. Spurious Flicker Ratio (SFR): Ratio of state transitions (Mudancas) to total anomaly count
    3. Spatial Coherence Proxy (SCP): Measure of spatial neighbor agreement
    """
    arr = df_pred.values # shape: (N_pixels, N_dates)
    N_pixels, N_dates = arr.shape
    
    # Identify anomaly positions (-1)
    is_anomaly = (arr == -1)
    total_anomalies = np.sum(is_anomaly)
    
    if total_anomalies == 0:
        return {'TPR': 0.0, 'SFR': 0.0, 'SCP': 0.0, 'Total_Anomalies': 0}
        
    # 1. Temporal Persistence Ratio (TPR)
    # Check if anomaly at t is also anomaly at t+1
    persistent_anomalies = np.sum(is_anomaly[:, :-1] & is_anomaly[:, 1:])
    tpr = persistent_anomalies / total_anomalies
    
    # 2. Spurious Flicker Ratio (SFR)
    arr_t = arr[:, 1:]
    arr_t_minus_1 = arr[:, :-1]
    total_transitions = np.sum((arr_t + arr_t_minus_1) == 0)
    sfr = total_transitions / total_anomalies
    
    # 3. Spatial Coherence Proxy (SCP)
    # Calculate agreement between neighboring pixels in index order
    # (Pixels close in dataframe index are geographically adjacent)
    spatial_diff = np.mean(arr[1:, :] == arr[:-1, :])
    scp = float(spatial_diff)
    
    return {
        'Total_Anomalies': int(total_anomalies),
        'Total_Transitions': int(total_transitions),
        'TPR': float(tpr),
        'SFR': float(sfr),
        'SCP': float(scp)
    }

def main():
    print("=== COMPUTING CUSTOM UNSUPERVISED METRICS FOR ALTAMIRA NDVI ===")
    dataset_name = 'Altamira'
    band_name = 'ndvi'
    
    # Load raw data
    df_raw = repro_utils.load_raw_data(dataset_name, band_name)
    if dataset_name == 'Altamira':
        qa_df = pd.read_parquet("data_Altamira_SummaryQA.parquet")
        good_dates = qa_df.columns[qa_df.mean(axis=0) < 1]
        df_raw = df_raw[good_dates]
        
    centered_vals = repro_utils.get_centered_data(df_raw, dataset_name, band_name, leak_free=True)
    
    # 1. Z-Score Baseline
    print("\n1. Evaluating Z-Score Baseline...")
    df_pred_base, _, _ = repro_utils.run_baseline_pipeline(df_raw, centered_vals, alpha=1.0, leak_free=True)
    metrics_base = compute_custom_unsupervised_metrics(df_pred_base)
    
    # 2. Isolation Forest (leak_free=True)
    print("\n2. Evaluating Isolation Forest (leak_free=True, n_est=40)...")
    df_pred_if, _, _ = repro_utils.run_experiment_pipeline(
        df_raw, centered_vals, alpha=1.0, beta=None, model_type='IsolationForest',
        model_params={'n_estimators': 40}, leak_free=True
    )
    metrics_if = compute_custom_unsupervised_metrics(df_pred_if)
    
    # 3. LSTM Autoencoder
    print("\n3. Evaluating LSTM Autoencoder...")
    matrix_path = f"Preprocess/{dataset_name}_{band_name.upper()}_CenteredMatrix.parquet"
    if not os.path.exists(matrix_path):
        repro_utils.run_and_log_preprocessing(dataset_name, band_name)
    df_mat = pd.read_parquet(matrix_path).replace(-99999, np.nan)
    
    feature_tensor = fe.build_time_aware_features(df_mat.values, window_size=3)
    feature_tensor = np.nan_to_num(feature_tensor, nan=0.0)
    
    x_tensor = torch.tensor(feature_tensor, dtype=torch.float32)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTMAutoencoder(num_features=5, hidden_dim=64, num_layers=2).to(device)
    
    # Evaluate reconstruction
    model.eval()
    dataloader = DataLoader(TensorDataset(x_tensor), batch_size=512, shuffle=False)
    reconstructed = []
    with torch.no_grad():
        for (batch_x,) in dataloader:
            rec = model(batch_x.to(device))
            reconstructed.append(rec.cpu().numpy())
    reconstructed = np.concatenate(reconstructed, axis=0)
    
    mse_errors = np.mean((feature_tensor - reconstructed) ** 2, axis=-1)
    threshold = np.percentile(mse_errors, 95)
    anomalies_mask = np.where(mse_errors > threshold, -1, 1)
    df_pred_lstm = pd.DataFrame(anomalies_mask, index=df_raw.index, columns=df_raw.columns)
    
    metrics_lstm = compute_custom_unsupervised_metrics(df_pred_lstm)
    
    print("\n=========================================================================")
    print("           CUSTOM UNSUPERVISED METRICS COMPARISON TABLE                  ")
    print("=========================================================================")
    print(f"{'Model':<25} | {'Total Anom':<12} | {'Flicker Ratio (SFR)':<20} | {'Persistence (TPR)':<18} | {'Spatial Coherence (SCP)'}")
    print("-" * 95)
    print(f"{'Z-Score Baseline':<25} | {metrics_base['Total_Anomalies']:<12} | {metrics_base['SFR']:<20.4f} | {metrics_base['TPR']:<18.4f} | {metrics_base['SCP']:.4f}")
    print(f"{'Isolation Forest':<25} | {metrics_if['Total_Anomalies']:<12} | {metrics_if['SFR']:<20.4f} | {metrics_if['TPR']:<18.4f} | {metrics_if['SCP']:.4f}")
    print(f"{'LSTM Autoencoder (Ours)':<25} | {metrics_lstm['Total_Anomalies']:<12} | {metrics_lstm['SFR']:<20.4f} | {metrics_lstm['TPR']:<18.4f} | {metrics_lstm['SCP']:.4f}")
    print("=========================================================================")

if __name__ == '__main__':
    main()
