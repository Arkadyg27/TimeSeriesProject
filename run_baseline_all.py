import repro_utils

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run Baseline Experiments")
    parser.add_argument('--suite', type=str, default='script', choices=['script', 'paper_text', 'all'],
                        help="Suite to run: 'script' (DynaLand), 'paper_text' (literal paper text), or 'all'")
    args = parser.parse_args()

    print(f"Starting Baseline Experiments (Suite: {args.suite.upper()})...")

    if args.suite == 'paper_text':
        datasets = [
            {'dataset': 'Altamira', 'band': 'ndvi', 'alpha': 1.0},
            {'dataset': 'Brumadinho_Landsat', 'band': 'ndvi', 'alpha': 1.0},
            {'dataset': 'Mariana_Sentinel', 'band': 'ndwi', 'alpha': 1.0}
        ]
    elif args.suite == 'script':
        datasets = [
            {'dataset': 'Altamira', 'band': 'ndvi', 'alpha': 1.0},
            {'dataset': 'Brumadinho', 'band': 'ndvi', 'alpha': 1.0},
            {'dataset': 'Brumadinho', 'band': 'ndwi', 'alpha': 1.0}, 
            {'dataset': 'Mariana', 'band': 'gvmi', 'alpha': 1.0},
            {'dataset': 'Mariana', 'band': 'ndwi', 'alpha': 1.0}
        ]
    else: # all
        datasets = [
            {'dataset': 'Altamira', 'band': 'ndvi', 'alpha': 1.0},
            {'dataset': 'Brumadinho', 'band': 'ndvi', 'alpha': 1.0},
            {'dataset': 'Brumadinho', 'band': 'ndwi', 'alpha': 1.0}, 
            {'dataset': 'Mariana', 'band': 'gvmi', 'alpha': 1.0},
            {'dataset': 'Mariana', 'band': 'ndwi', 'alpha': 1.0},
            {'dataset': 'Brumadinho_Landsat', 'band': 'ndvi', 'alpha': 1.0},
            {'dataset': 'Mariana_Sentinel', 'band': 'ndwi', 'alpha': 1.0}
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

    print(f"\nAll Baseline experiments for {args.suite.upper()} suite logged successfully to MLflow!")

if __name__ == '__main__':
    main()
