import repro_utils

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run Preprocessing Pipelines")
    parser.add_argument('--suite', type=str, default='all', choices=['script', 'paper_text', 'all'],
                        help="Suite to preprocess: 'script' (DynaLand), 'paper_text' (literal paper text), or 'all'")
    parser.add_argument('--alpha', type=float, default=0.5, help="Envelope threshold multiplier alpha")
    args = parser.parse_args()

    print(f"Starting baseline data ingestion, cloud filtering, centering, and envelope selection (Suite: {args.suite.upper()})...")
    alpha = args.alpha
    
    if args.suite == 'paper_text':
        tasks = [
            ('Altamira', 'ndvi'),
            ('Brumadinho_Landsat', 'ndvi'),
            ('Mariana_Sentinel', 'ndwi')
        ]
    elif args.suite == 'script':
        tasks = [
            ('Altamira', 'ndvi'),
            ('Brumadinho', 'ndwi'),
            ('Mariana', 'gvmi'),
            ('Brumadinho', 'ndvi'),
            ('Mariana', 'ndwi')
        ]
    else: # all
        tasks = [
            ('Altamira', 'ndvi'),
            ('Brumadinho', 'ndwi'),
            ('Mariana', 'gvmi'),
            ('Brumadinho', 'ndvi'),
            ('Mariana', 'ndwi'),
            ('Brumadinho_Landsat', 'ndvi'),
            ('Mariana_Sentinel', 'ndwi')
        ]
    
    for ds_name, band in tasks:
        repro_utils.run_and_log_preprocessing(ds_name, band, alpha=alpha)
    
    print(f"\nAll preprocessing pipelines for {args.suite.upper()} executed and logged to MLflow successfully!")

if __name__ == '__main__':
    main()
