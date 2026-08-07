import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import time
import os
import matplotlib.pyplot as plt
import mlflow
from torch.utils.data import TensorDataset, DataLoader
import feature_engineering as fe
import repro_utils
from models_deep import LSTMAutoencoder

def train_lstm_autoencoder(dataset_name='Altamira', index_name='NDVI', batch_size=512, lr=0.001):
    print(f"==================== Deep Learning: {dataset_name} {index_name} (LSTM Autoencoder) ====================")
    print("Loading precomputed centered matrix...")
    matrix_path = f"Preprocess/{dataset_name}_{index_name}_CenteredMatrix.parquet"
    if not os.path.exists(matrix_path):
        print(f"Centered matrix {matrix_path} not found. Running preprocessing...")
        repro_utils.run_and_log_preprocessing(dataset_name, index_name.lower())
        
    df = pd.read_parquet(matrix_path)
    df = df.replace(-99999, np.nan)
    
    print("Extracting Time-Aware Features (Velocity, Acceleration, Rolling Stats)...")
    feature_tensor = fe.build_time_aware_features(df.values, window_size=3)
    feature_tensor = np.nan_to_num(feature_tensor, nan=0.0)
    
    # Convert numpy array to PyTorch float32 tensor
    x_tensor = torch.tensor(feature_tensor, dtype=torch.float32)
    
    # Create DataLoader
    dataset = TensorDataset(x_tensor, x_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- USING DEVICE: {device} ---")
    
    model = LSTMAutoencoder(num_features=5, hidden_dim=64, num_layers=2).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Setup MLflow
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    if os.environ.get("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(f"DeepLearning_{dataset_name}_{index_name}_LSTM_AE")
    
    t_start = time.time()
    
    epoch_targets = [10, 20]
    percentiles = [90, 95, 99]
    
    current_epoch = 0
    
    for target_epochs in epoch_targets:
        print(f"Training up to {target_epochs} epochs...")
        model.train()
        avg_loss = 0.0
        
        for epoch in range(current_epoch, target_epochs):
            epoch_loss = 0.0
            t0 = time.time()
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                
                optimizer.zero_grad()
                outputs = model(batch_x)
                
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                
            avg_loss = epoch_loss / len(dataloader)
            t1 = time.time()
            print(f"Epoch [{epoch+1}/{target_epochs}], Loss: {avg_loss:.6f}, Time: {t1-t0:.2f}s")
            
        current_epoch = target_epochs
        
        print(f"Evaluating reconstruction errors at Epoch {target_epochs}...")
        model.eval()
        reconstructed = []
        eval_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        with torch.no_grad():
            for batch_x, _ in eval_dataloader:
                batch_x = batch_x.to(device)
                rec = model(batch_x)
                reconstructed.append(rec.cpu().numpy())
                
        reconstructed = np.concatenate(reconstructed, axis=0)
        
        # Calculate MSE per pixel and date
        mse_errors = np.mean((feature_tensor - reconstructed) ** 2, axis=-1)
        
        for pct in percentiles:
            run_name = f"Ep{target_epochs}_Pct{pct}"
            print(f"Generating MLflow run for {run_name}...")
            
            with mlflow.start_run(run_name=run_name):
                mlflow.log_params({
                    "model": "LSTMAutoencoder",
                    "epochs": target_epochs,
                    "batch_size": batch_size,
                    "learning_rate": lr,
                    "hidden_dim": 64,
                    "percentile_threshold": pct
                })
                
                mlflow.log_metric("final_train_loss", avg_loss)
                
                anomaly_threshold = np.percentile(mse_errors, pct)
                anomalies_mask = np.where(mse_errors > anomaly_threshold, -1, 1)
                
                df_pred = pd.DataFrame(anomalies_mask, index=df.index, columns=df.columns)
                
                # 2. Calculate Transition Metrics
                df_metrics = repro_utils.calculate_metrics(df_pred)
                
                t_end = time.time()
                exec_time = t_end - t_start
                
                # Log Summary Metrics
                total_anomalies = int(df_metrics['Anomalias'].sum())
                total_regular = int(df_metrics['Regular'].sum())
                total_transitions = int(df_metrics['Mudanças'].sum())
                mean_transitions = float(df_metrics['media'].iloc[0])
                std_transitions = float(df_metrics['std'].iloc[0])

                mlflow.log_metric("total_anomalies", total_anomalies)
                mlflow.log_metric("total_regular", total_regular)
                mlflow.log_metric("total_transitions", total_transitions)
                mlflow.log_metric("mean_transitions", mean_transitions)
                mlflow.log_metric("std_transitions", std_transitions)
                mlflow.log_metric("execution_time_seconds", exec_time)
                mlflow.log_metric("anomaly_threshold", anomaly_threshold)
                
                # Log custom unsupervised metrics (SCP, SFR, TPR, Cluster Size, Entropy)
                custom_m = repro_utils.compute_summary_unsupervised_metrics(df_pred)
                mlflow.log_metric("flicker_ratio_sfr", custom_m['flicker_ratio_sfr'])
                mlflow.log_metric("temporal_persistence_tpr", custom_m['temporal_persistence_tpr'])
                mlflow.log_metric("spatial_coherence_scp", custom_m['spatial_coherence_scp'])
                mlflow.log_metric("avg_cluster_size", custom_m['avg_cluster_size'])
                mlflow.log_metric("temporal_entropy_h", custom_m['temporal_entropy_h'])

                
                # 3. Save GeoTIFF Artifact
                tiff_dir = f"Tiff/deep_learning/{dataset_name}_{index_name}"
                os.makedirs(tiff_dir, exist_ok=True)
                tiff_path = os.path.join(tiff_dir, f"{run_name}_LSTM_Autoencoder.tif")
                repro_utils.save_tiff_fromdf(df_metrics, ['Anomalias', 'p-valor'], -99999, tiff_path)
                mlflow.log_artifact(tiff_path)
                
                # 4. Generate & Save Spatial Anomaly Map PNG Figure
                plt.figure(figsize=(10, 8), dpi=200)
                lats = df.index.get_level_values('Latitude')
                lons = df.index.get_level_values('Longitude')
                sc = plt.scatter(lons, lats, c=df_metrics['Anomalias'], cmap='hot', s=4)
                plt.colorbar(sc, label='Anomaly Count')
                plt.title(f"{dataset_name} {index_name} - LSTM Anomaly Map ({run_name})")
                plt.xlabel("Longitude")
                plt.ylabel("Latitude")
                
                fig_path = f"{dataset_name}_{index_name}_{run_name}_LSTM_Anomaly_Map.png"
                plt.savefig(fig_path)
                plt.close()
                mlflow.log_artifact(fig_path)
                
                # Save model weights for this epoch
                model_path = f"lstm_autoencoder_{run_name}.pth"
                torch.save(model.state_dict(), model_path)
                mlflow.log_artifact(model_path)
                
                print(f"Logged {run_name} successfully!")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Train LSTM Autoencoder on Spatio-Temporal Datasets")
    parser.add_argument('--suite', type=str, default='all', choices=['all', 'script', 'paper_text'],
                        help="Dataset suite to train: 'script' (DynaLand codebase), 'paper_text' (literal paper text), or 'all'")
    parser.add_argument('--dataset', type=str, default=None, help="Specific dataset name (e.g. Brumadinho_Landsat)")
    parser.add_argument('--index', type=str, default=None, help="Specific index name (e.g. NDVI)")
    parser.add_argument('--batch_size', type=int, default=512, help="Batch size for training and evaluation")
    args = parser.parse_args()

    if args.dataset and args.index:
        train_lstm_autoencoder(dataset_name=args.dataset, index_name=args.index, batch_size=args.batch_size)
    elif args.suite == 'paper_text':
        paper_text_combinations = [
            ('Altamira', 'NDVI'),
            ('Brumadinho_Landsat', 'NDVI'),
            ('Mariana_Sentinel', 'NDWI')
        ]
        for dataset, index in paper_text_combinations:
            train_lstm_autoencoder(dataset_name=dataset, index_name=index, batch_size=args.batch_size)
    elif args.suite == 'script':
        script_combinations = [
            ('Altamira', 'NDVI'),
            ('Brumadinho', 'NDVI'),
            ('Brumadinho', 'NDWI'),
            ('Mariana', 'NDWI'),
            ('Mariana', 'GVMI')
        ]
        for dataset, index in script_combinations:
            train_lstm_autoencoder(dataset_name=dataset, index_name=index, batch_size=args.batch_size)
    else: # 'all'
        all_combinations = [
            ('Altamira', 'NDVI'),
            ('Brumadinho', 'NDVI'),
            ('Brumadinho', 'NDWI'),
            ('Mariana', 'NDWI'),
            ('Mariana', 'GVMI'),
            ('Brumadinho_Landsat', 'NDVI'),
            ('Mariana_Sentinel', 'NDWI')
        ]
        for dataset, index in all_combinations:
            train_lstm_autoencoder(dataset_name=dataset, index_name=index, batch_size=args.batch_size)
