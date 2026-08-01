import os
import glob
import numpy as np
import pandas as pd
import rasterio
import mlflow
import repro_utils

def compute_metrics_from_tiff(tiff_path):
    """
    Computes unsupervised metrics directly from saved output GeoTIFF files
    without needing to re-run model training or data fetching.
    """
    with rasterio.open(tiff_path) as src:
        # Band 1: Anomalies (-1 for anomaly, 1 for regular)
        arr = src.read(1)
        
    # Replace nodata (-99999) with NaN
    valid_mask = (arr != -99999) & ~np.isnan(arr)
    
    # Identify anomalies (-1)
    is_anomaly = (arr == -1) & valid_mask
    total_anomalies = int(np.sum(is_anomaly))
    
    if total_anomalies == 0:
        return {
            'flicker_ratio_sfr': 0.0,
            'temporal_persistence_tpr': 0.0,
            'spatial_coherence_scp': 0.0,
            'avg_cluster_size': 0.0,
            'temporal_entropy_h': 0.0,
            'total_anomalies': 0
        }
        
    # 1. Spatial Coherence Index (SCP): adjacent pixel agreement
    arr_valid = np.where(valid_mask, arr, np.nan)
    horizontal_match = np.mean(arr_valid[:, :-1] == arr_valid[:, 1:])
    vertical_match = np.mean(arr_valid[:-1, :] == arr_valid[1:, :])
    scp = float(np.nanmean([horizontal_match, vertical_match]))
    
    # 2. Average Spatial Cluster Size (connected components of anomalies)
    from scipy.ndimage import label
    labeled_array, num_features = label(is_anomaly)
    avg_cluster = float(total_anomalies / num_features) if num_features > 0 else 0.0
    
    # 3. Temporal / Spatial Entropy proxy
    p = total_anomalies / np.sum(valid_mask) if np.sum(valid_mask) > 0 else 0.0
    p_clipped = np.clip(p, 1e-7, 1 - 1e-7)
    entropy_h = float(-p_clipped * np.log2(p_clipped) - (1 - p_clipped) * np.log2(1 - p_clipped))
    
    return {
        'flicker_ratio_sfr': 0.0, # N/A for single 2D spatial raster aggregation
        'temporal_persistence_tpr': 0.0,
        'spatial_coherence_scp': float(scp),
        'avg_cluster_size': float(avg_cluster),
        'temporal_entropy_h': float(entropy_h),
        'total_anomalies': total_anomalies
    }

def main():
    print("=== COMPUTING METRICS FOR ALL EXISTING TIFFF RESULTS ===")
    tiff_files = glob.glob("Tiff/**/*.tif", recursive=True)
    print(f"Found {len(tiff_files)} output GeoTIFF files.")
    
    results = []
    for tf in tiff_files:
        basename = os.path.basename(tf)
        metrics = compute_metrics_from_tiff(tf)
        results.append({
            'File': basename,
            'Protocol': 'leak_free' if 'leak_free' in tf else ('leaky' if 'leaky' in tf else 'deep_learning'),
            'Total Anomalies': metrics['total_anomalies'],
            'Spatial Coherence (SCP)': f"{metrics['spatial_coherence_scp']:.4f}",
            'Avg Cluster Size (px)': f"{metrics['avg_cluster_size']:.2f}",
            'Entropy (H)': f"{metrics['temporal_entropy_h']:.4f}"
        })
        
    df_res = pd.DataFrame(results)
    print("\n=========================================================================")
    print("             EXISTING EXPERIMENT RESULTS METRICS TABLE                   ")
    print("=========================================================================")
    print(df_res.to_string(index=False))
    print("=========================================================================")

if __name__ == '__main__':
    main()
