import repro_utils

def main():
    print("Starting Baseline Experiments for all datasets...")

    # Define the datasets and their respective bands and envelope multiplier (alpha)
    datasets = [
        {'dataset': 'Altamira', 'band': 'ndvi', 'alpha': 1.0},
        {'dataset': 'Brumadinho', 'band': 'ndvi', 'alpha': 1.0},
        {'dataset': 'Brumadinho', 'band': 'ndwi', 'alpha': 1.0}, 
        {'dataset': 'Mariana', 'band': 'gvmi', 'alpha': 1.0},
        {'dataset': 'Mariana', 'band': 'ndwi', 'alpha': 1.0}
    ]
    
    for ds in datasets:
        dataset_name = ds['dataset']
        band_name = ds['band']
        alpha = ds['alpha']
        
        print(f"\n=======================================================")
        print(f"Running Baseline for {dataset_name} ({band_name.upper()}) with alpha={alpha}")
        print(f"=======================================================")
        
        # Run both the leaky and the walk-forward (leak_free) baselines
        for leak_free in [False, True]:
            print(f"\n--- Running Z-Score Baseline (leak_free={leak_free}) ---")
            repro_utils.log_baseline_to_mlflow(
                dataset_name=dataset_name,
                band_name=band_name,
                alpha=alpha,
                leak_free=leak_free
            )

    print("\nAll Baseline experiments logged successfully to MLflow!")

if __name__ == '__main__':
    main()
