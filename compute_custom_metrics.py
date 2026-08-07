import os
import time
import numpy as np
import pandas as pd
import torch
import mlflow
from torch.utils.data import TensorDataset, DataLoader
import repro_utils
import feature_engineering as fe
from models_deep import LSTMAutoencoder

EVENT_DATES = {
    'Altamira': '2016-01-01',
    'Brumadinho': '2019-01-25',
    'Mariana': '2015-11-05'
}

def compute_full_unsupervised_metrics_suite(df_pred, exec_time, dataset_name):
    """
    Computes and logs the complete suite of unsupervised spatial-temporal metrics:
    1. Spatial Coherence Index (SCP)
    2. Spurious Flicker Ratio (SFR)
    3. Temporal Persistence Ratio (TPR)
    4. Average Spatial Cluster Size (S_cluster)
    5. Temporal Prediction Entropy (Entropy H)
    6. Disaster Event Contrast Ratio (CNR)
    7. Execution Time & Inference Speed (ms/sample)
    """
    arr = df_pred.values # (N_pixels, N_dates)
    N_pixels, N_dates = arr.shape
    
    is_anomaly = (arr == -1)
    total_anomalies = int(np.sum(is_anomaly))
    
    if total_anomalies == 0:
        return {
            'spatial_coherence_scp': 0.0,
            'flicker_ratio_sfr': 0.0,
            'temporal_persistence_tpr': 0.0,
            'avg_cluster_size': 0.0,
            'temporal_entropy_h': 0.0,
            'disaster_contrast_cnr': 1.0,
            'execution_time_seconds': float(exec_time),
            'inference_speed_ms_per_pixel': float((exec_time * 1000) / N_pixels)
        }
        
    # 1. Temporal Persistence Ratio (TPR)
    persistent_anomalies = np.sum(is_anomaly[:, :-1] & is_anomaly[:, 1:])
    tpr = float(persistent_anomalies / total_anomalies) if total_anomalies > 0 else 0.0
    
    # 2. Spurious Flicker Ratio (SFR)
    arr_t = arr[:, 1:]
    arr_t_minus_1 = arr[:, :-1]
    total_transitions = np.sum((arr_t + arr_t_minus_1) == 0)
    sfr = float(total_transitions / total_anomalies) if total_anomalies > 0 else 0.0
    
    # 3. Spatial Coherence Index (SCP)
    scp = float(np.mean(arr[1:, :] == arr[:-1, :]))
    
    # 4. Temporal Prediction Entropy (H)
    p = np.mean(is_anomaly, axis=1)
    p_clipped = np.clip(p, 1e-7, 1 - 1e-7)
    h_pixel = -p_clipped * np.log2(p_clipped) - (1 - p_clipped) * np.log2(1 - p_clipped)
    entropy_h = float(np.mean(h_pixel))
    
    # 5. Average Spatial Cluster Size (S_cluster)
    from scipy.ndimage import label
    try:
        lat = [idx[0] for idx in df_pred.index]
        lon = [idx[1] for idx in df_pred.index]
        ulat = np.unique(lat)
        ulon = np.unique(lon)
        ncols = len(ulon)
        nrows = len(ulat)
        
        ys = ulat[1] - ulat[0] if len(ulat) > 1 else (ulat[11] - ulat[10] if len(ulat) > 11 else 0.002)
        xs = ulon[1] - ulon[0] if len(ulon) > 1 else (ulon[11] - ulon[10] if len(ulon) > 11 else 0.002)
        refLat = np.max(ulat)
        refLon = np.min(ulon)
        
        anom_map_2d = np.zeros((nrows, ncols), dtype=bool)
        anom_counts = np.sum(is_anomaly, axis=1)
        
        for j in range(len(df_pred)):
            if anom_counts[j] > 0:
                posLin = np.clip(np.int64(np.round((refLat - lat[j]) / abs(ys))), 0, nrows - 1)
                posCol = np.clip(np.int64(np.round((lon[j] - refLon) / abs(xs))), 0, ncols - 1)
                anom_map_2d[posLin, posCol] = True
                
        labeled_array, num_features = label(anom_map_2d)
        avg_cluster = float(np.sum(anom_map_2d) / num_features) if num_features > 0 else 0.0
    except Exception:
        avg_cluster = 0.0
        
    # 6. Disaster Event Contrast Ratio (CNR)
    event_date = EVENT_DATES.get(dataset_name, '2019-01-01')
    event_mask = np.array([str(col) >= event_date for col in df_pred.columns])
    
    if np.sum(event_mask) > 0 and np.sum(~event_mask) > 0:
        event_anomaly_density = np.mean(is_anomaly[:, event_mask])
        baseline_anomaly_density = np.mean(is_anomaly[:, ~event_mask])
        cnr = float(event_anomaly_density / (baseline_anomaly_density + 1e-6))
    else:
        cnr = 1.0
        
    # 7. Execution Time & Inference Speed
    speed_ms_per_pixel = float((exec_time * 1000) / N_pixels)
    
    return {
        'total_anomalies': int(total_anomalies),
        'total_regular': int(np.sum(~is_anomaly)),
        'spatial_coherence_scp': float(scp),
        'flicker_ratio_sfr': float(sfr),
        'temporal_persistence_tpr': float(tpr),
        'avg_cluster_size': float(avg_cluster),
        'temporal_entropy_h': float(entropy_h),
        'disaster_contrast_cnr': float(cnr),
        'execution_time_seconds': float(exec_time),
        'inference_speed_ms_per_pixel': float(speed_ms_per_pixel)
    }


def evaluate_dataset(dataset_name, band_name):
    print(f"\n==================== EVALUATING DATASET: {dataset_name} ({band_name.upper()}) ====================")
    df_raw = repro_utils.load_raw_data(dataset_name, band_name)
    if dataset_name == 'Altamira':
        qa_df = pd.read_parquet("data_Altamira_SummaryQA.parquet")
        good_dates = qa_df.columns[qa_df.mean(axis=0) < 1]
        df_raw = df_raw[good_dates]
        
    centered_vals = repro_utils.get_centered_data(df_raw, dataset_name, band_name, leak_free=True)
    
    # Setup MLflow Experiment safely
    repro_utils.safe_set_experiment(f"Unsupervised_Full_Suite_{dataset_name}")
    
    results = []
    
    # 1. Z-Score Baseline
    print("1. Evaluating Z-Score Baseline...")
    df_pred_base, _, exec_time_base = repro_utils.run_baseline_pipeline(df_raw, centered_vals, alpha=1.0, leak_free=True)
    m_base = compute_full_unsupervised_metrics_suite(df_pred_base, exec_time_base, dataset_name)
    
    with mlflow.start_run(run_name=f"{dataset_name}_ZScore_Baseline"):
        mlflow.log_params({"dataset": dataset_name, "band": band_name, "model": "Z-Score Baseline"})
        mlflow.log_metrics(m_base)
        
    results.append({'Dataset': dataset_name, 'Model': 'Z-Score Baseline', 'SCP': f"{m_base['spatial_coherence_scp']:.4f}", 'SFR': f"{m_base['flicker_ratio_sfr']:.4f}", 'TPR': f"{m_base['temporal_persistence_tpr']:.4f}", 'Cluster Size': f"{m_base['avg_cluster_size']:.2f}", 'Entropy H': f"{m_base['temporal_entropy_h']:.4f}", 'CNR Event': f"{m_base['disaster_contrast_cnr']:.2f}x", 'Exec Time (s)': f"{m_base['execution_time_seconds']:.1f}s", 'Speed (ms/px)': f"{m_base['inference_speed_ms_per_pixel']:.3f}ms"})
    
    # 2. Isolation Forest (leak_free=True)
    print("2. Evaluating Isolation Forest (leak_free=True)...")
    df_pred_if, _, exec_time_if = repro_utils.run_experiment_pipeline(
        df_raw, centered_vals, alpha=1.0, beta=None, model_type='IsolationForest',
        model_params={'n_estimators': 40}, leak_free=True
    )
    m_if = compute_full_unsupervised_metrics_suite(df_pred_if, exec_time_if, dataset_name)
    
    with mlflow.start_run(run_name=f"{dataset_name}_Isolation_Forest"):
        mlflow.log_params({"dataset": dataset_name, "band": band_name, "model": "Isolation Forest", "n_estimators": 40})
        mlflow.log_metrics(m_if)
        
    results.append({'Dataset': dataset_name, 'Model': 'Isolation Forest', 'SCP': f"{m_if['spatial_coherence_scp']:.4f}", 'SFR': f"{m_if['flicker_ratio_sfr']:.4f}", 'TPR': f"{m_if['temporal_persistence_tpr']:.4f}", 'Cluster Size': f"{m_if['avg_cluster_size']:.2f}", 'Entropy H': f"{m_if['temporal_entropy_h']:.4f}", 'CNR Event': f"{m_if['disaster_contrast_cnr']:.2f}x", 'Exec Time (s)': f"{m_if['execution_time_seconds']:.1f}s", 'Speed (ms/px)': f"{m_if['inference_speed_ms_per_pixel']:.3f}ms"})

    # 3. One-Class SVM (leak_free=True, default parameters, no sweep)
    print("3. Evaluating One-Class SVM (leak_free=True)...")
    df_pred_ocsvm, _, exec_time_ocsvm = repro_utils.run_experiment_pipeline(
        df_raw, centered_vals, alpha=1.0, beta=None, model_type='OCSVM',
        model_params={'nu': 0.05, 'gamma': 'scale'}, leak_free=True
    )
    m_ocsvm = compute_full_unsupervised_metrics_suite(df_pred_ocsvm, exec_time_ocsvm, dataset_name)
    
    with mlflow.start_run(run_name=f"{dataset_name}_OneClass_SVM"):
        mlflow.log_params({"dataset": dataset_name, "band": band_name, "model": "One-Class SVM", "nu": 0.05, "gamma": "scale"})
        mlflow.log_metrics(m_ocsvm)
        
    results.append({'Dataset': dataset_name, 'Model': 'One-Class SVM', 'SCP': f"{m_ocsvm['spatial_coherence_scp']:.4f}", 'SFR': f"{m_ocsvm['flicker_ratio_sfr']:.4f}", 'TPR': f"{m_ocsvm['temporal_persistence_tpr']:.4f}", 'Cluster Size': f"{m_ocsvm['avg_cluster_size']:.2f}", 'Entropy H': f"{m_ocsvm['temporal_entropy_h']:.4f}", 'CNR Event': f"{m_ocsvm['disaster_contrast_cnr']:.2f}x", 'Exec Time (s)': f"{m_ocsvm['execution_time_seconds']:.1f}s", 'Speed (ms/px)': f"{m_ocsvm['inference_speed_ms_per_pixel']:.3f}ms"})

    # 4. LSTM Autoencoder
    print("4. Evaluating LSTM Autoencoder...")
    matrix_path = f"Preprocess/{dataset_name}_{band_name.upper()}_CenteredMatrix.parquet"
    if not os.path.exists(matrix_path):
        repro_utils.run_and_log_preprocessing(dataset_name, band_name)
    df_mat = pd.read_parquet(matrix_path).replace(-99999, np.nan)
    
    t0_lstm = time.time()
    feature_tensor = fe.build_time_aware_features(df_mat.values, window_size=3)
    feature_tensor = np.nan_to_num(feature_tensor, nan=0.0)
    
    x_tensor = torch.tensor(feature_tensor, dtype=torch.float32)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTMAutoencoder(num_features=5, hidden_dim=64, num_layers=2).to(device)
    
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
    exec_time_lstm = time.time() - t0_lstm
    
    m_lstm = compute_full_unsupervised_metrics_suite(df_pred_lstm, exec_time_lstm, dataset_name)
    
    with mlflow.start_run(run_name=f"{dataset_name}_LSTM_Autoencoder"):
        mlflow.log_params({"dataset": dataset_name, "band": band_name, "model": "LSTM Autoencoder"})
        mlflow.log_metrics(m_lstm)
        
    results.append({'Dataset': dataset_name, 'Model': 'LSTM Autoencoder (Ours)', 'SCP': f"{m_lstm['spatial_coherence_scp']:.4f}", 'SFR': f"{m_lstm['flicker_ratio_sfr']:.4f}", 'TPR': f"{m_lstm['temporal_persistence_tpr']:.4f}", 'Cluster Size': f"{m_lstm['avg_cluster_size']:.2f}", 'Entropy H': f"{m_lstm['temporal_entropy_h']:.4f}", 'CNR Event': f"{m_lstm['disaster_contrast_cnr']:.2f}x", 'Exec Time (s)': f"{m_lstm['execution_time_seconds']:.1f}s", 'Speed (ms/px)': f"{m_lstm['inference_speed_ms_per_pixel']:.3f}ms"})

    return results

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Compute Full Unsupervised Metrics Suite")
    parser.add_argument('--suite', type=str, default='script', choices=['script', 'paper_text', 'all'],
                        help="Dataset suite to evaluate: 'script' (DynaLand), 'paper_text' (literal paper text), or 'all'")
    args = parser.parse_args()

    print(f"=== LOGGING FULL UNSUPERVISED METRICS SUITE TO MLFLOW (SUITE: {args.suite.upper()}) ===")
    
    if args.suite == 'paper_text':
        datasets = [
            {'name': 'Altamira', 'band': 'ndvi'},
            {'name': 'Brumadinho_Landsat', 'band': 'ndvi'},
            {'name': 'Mariana_Sentinel', 'band': 'ndwi'}
        ]
        out_csv = "PAPER_TEXT_ALL_METRICS.csv"
        out_md = "PAPER_TEXT_ALL_METRICS.md"
    elif args.suite == 'script':
        datasets = [
            {'name': 'Altamira', 'band': 'ndvi'},
            {'name': 'Brumadinho', 'band': 'ndwi'},
            {'name': 'Mariana', 'band': 'gvmi'}
        ]
        out_csv = "THREE_DATASETS_ALL_METRICS.csv"
        out_md = "THREE_DATASETS_ALL_METRICS.md"
    else: # all
        datasets = [
            {'name': 'Altamira', 'band': 'ndvi'},
            {'name': 'Brumadinho', 'band': 'ndwi'},
            {'name': 'Mariana', 'band': 'gvmi'},
            {'name': 'Brumadinho_Landsat', 'band': 'ndvi'},
            {'name': 'Mariana_Sentinel', 'band': 'ndwi'}
        ]
        out_csv = "ALL_SUITES_ALL_METRICS.csv"
        out_md = "ALL_SUITES_ALL_METRICS.md"
    
    all_results = []
    for item in datasets:
        res = evaluate_dataset(item['name'], item['band'])
        all_results.extend(res)
        
    df_res = pd.DataFrame(all_results)
    
    print("\n=====================================================================================================================================================")
    print("                                                 COMPLETE UNSUPERVISED METRICS SUITE TABLE                                                          ")
    print("=====================================================================================================================================================")
    print(df_res.to_string(index=False))
    print("=====================================================================================================================================================")
    
    # Save CSV and MD
    df_res.to_csv(out_csv, index=False)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(f"# Complete Unsupervised Metrics Suite ({args.suite.upper()} Configuration)\n\n")
        f.write(df_res.to_markdown(index=False))
        f.write("\n")
    print(f"\nSaved metrics summary to {out_csv} and {out_md}!")

if __name__ == '__main__':
    main()
