import repro_utils
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

def run_preprocessing_no_mlflow(dataset_name, band_name, alpha=0.5):
    print(f"\n==================== Preprocessing: {dataset_name} ({band_name.upper()}) ====================")
    df_raw = repro_utils.load_raw_data(dataset_name, band_name)
    
    total_raw_dates = df_raw.shape[1]
    if dataset_name == 'Altamira':
        qa_df = pd.read_parquet("data_Altamira_SummaryQA.parquet")
        good_dates = qa_df.columns[qa_df.mean(axis=0) < 1]
        df_raw = df_raw[good_dates]
    
    unmasked_dates = df_raw.shape[1]
    print(f"Cloud/quality filtering: kept {unmasked_dates}/{total_raw_dates} dates.")
    
    default_dummy = -99999
    df_raw_nan = df_raw.replace(default_dummy, np.nan)
    raw_vals = df_raw_nan.values
    N_pixels, N_dates = raw_vals.shape
    
    trend_image = np.nanmedian(raw_vals, axis=1)
    centered_vals = raw_vals - trend_image[:, np.newaxis]
    
    all_vals = centered_vals.flatten()
    all_vals = all_vals[~np.isnan(all_vals)]
    
    mean_all = np.mean(all_vals)
    sigma = np.std(all_vals)
    
    inf_lim = mean_all - alpha * sigma
    sup_lim = mean_all + alpha * sigma
    regular_data = all_vals[(all_vals > inf_lim) & (all_vals < sup_lim)]
    
    # Save Trend Image as TIFF
    trend_df = pd.DataFrame(index=df_raw.index)
    trend_df['Trend_Median'] = trend_image
    trend_tiff_path = f"Preprocess/{dataset_name}_{band_name.upper()}_TrendImage.tif"
    os.makedirs(os.path.dirname(trend_tiff_path), exist_ok=True)
    repro_utils.save_tiff_fromdf(trend_df, ['Trend_Median'], default_dummy, trend_tiff_path)
    
    # Save Centered Matrix to Parquet
    centered_df = pd.DataFrame(centered_vals, index=df_raw.index, columns=df_raw.columns)
    centered_df = centered_df.fillna(default_dummy)
    centered_parquet_path = f"Preprocess/{dataset_name}_{band_name.upper()}_CenteredMatrix.parquet"
    centered_df.to_parquet(centered_parquet_path)
    
    # Save Stats
    stats_path = f"Preprocess/{dataset_name}_{band_name.upper()}_TrendImage_Stats.txt"
    with open(stats_path, 'w') as f_stats:
        f_stats.write(f"Dataset: {dataset_name}\n")
        f_stats.write(f"Band: {band_name}\n")
        f_stats.write(f"Trend Image Mean: {np.nanmean(trend_image):.6f}\n")
        f_stats.write(f"Trend Image Std Dev: {np.nanstd(trend_image):.6f}\n")
        f_stats.write(f"Trend Image Min: {np.nanmin(trend_image):.6f}\n")
        f_stats.write(f"Trend Image Max: {np.nanmax(trend_image):.6f}\n")
        f_stats.write(f"Trend Image Median: {np.nanmedian(trend_image):.6f}\n")
        
    print(f"Finished Preprocessing for {dataset_name}. Global Std: {sigma:.4f}, Regular count: {len(regular_data)}")

def main():
    alpha = 0.5
    print("Running missing datasets without MLflow...")
    run_preprocessing_no_mlflow('Brumadinho', 'ndvi', alpha=alpha)
    run_preprocessing_no_mlflow('Mariana', 'ndwi', alpha=alpha)
    print("Done!")

if __name__ == '__main__':
    main()
