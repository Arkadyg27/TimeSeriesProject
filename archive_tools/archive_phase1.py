import os
import shutil
import glob

def archive_current_results():
    base_dir = os.path.abspath(".")
    archive_dir = os.path.join(base_dir, "results_archive_script_dataset")
    
    # Target folders
    folders = {
        "data": os.path.join(archive_dir, "data"),
        "Preprocess": os.path.join(archive_dir, "Preprocess"),
        "plots": os.path.join(archive_dir, "plots"),
        "Tiff": os.path.join(archive_dir, "Tiff"),
        "metrics": os.path.join(archive_dir, "metrics")
    }
    
    for f in folders.values():
        os.makedirs(f, exist_ok=True)
        
    # 1. Copy raw parquet files
    parquet_files = glob.glob(os.path.join(base_dir, "data_*.parquet"))
    for pf in parquet_files:
        shutil.copy2(pf, folders["data"])
        print(f"Archived: {os.path.basename(pf)} -> data/")
        
    # 2. Copy Preprocess folder
    prep_dir = os.path.join(base_dir, "Preprocess")
    if os.path.exists(prep_dir):
        for item in os.listdir(prep_dir):
            s = os.path.join(prep_dir, item)
            d = os.path.join(folders["Preprocess"], item)
            if os.path.isfile(s):
                shutil.copy2(s, d)
        print("Archived Preprocess/ directory.")
        
    # 3. Copy plots
    plots_dir = os.path.join(base_dir, "plots")
    if os.path.exists(plots_dir):
        for item in os.listdir(plots_dir):
            s = os.path.join(plots_dir, item)
            d = os.path.join(folders["plots"], item)
            if os.path.isfile(s):
                shutil.copy2(s, d)
        print("Archived plots/ directory.")
        
    # 4. Copy Tiff
    tiff_dir = os.path.join(base_dir, "Tiff")
    if os.path.exists(tiff_dir):
        for item in os.listdir(tiff_dir):
            s = os.path.join(tiff_dir, item)
            d = os.path.join(folders["Tiff"], item)
            if os.path.isfile(s):
                shutil.copy2(s, d)
        print("Archived Tiff/ directory.")
        
    # 5. Copy metrics & reports
    metric_files = [
        "THREE_DATASETS_ALL_METRICS.csv",
        "THREE_DATASETS_ALL_METRICS.md",
        "FULL_RESULTS.md",
        "full_tiff_results.csv",
        "Template_Final_Report.md"
    ]
    for mf in metric_files:
        src = os.path.join(base_dir, mf)
        if os.path.exists(src):
            shutil.copy2(src, folders["metrics"])
            shutil.copy2(src, archive_dir)
            print(f"Archived metric file: {mf}")

    # Add a README to the archive
    readme_path = os.path.join(archive_dir, "README_ARCHIVE.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write("# Archive: Script Dataset Experimental Results\n\n")
        f.write("This archive contains the snapshot of all raw data, preprocessed centered matrices, ")
        f.write("trained model outputs, GeoTIFFs, plots, and metric benchmarks evaluated on the ")
        f.write("author's reference codebase (`DynaLand` scripts) configuration:\n\n")
        f.write("- **Altamira**: Terra MODIS (250m) NDVI\n")
        f.write("- **Brumadinho**: Sentinel-2 (10m) NDWI & NDVI\n")
        f.write("- **Mariana**: Landsat-8 (30m) GVMI & NDWI\n")
        
    print("\nPhase 1 Archiving complete successfully!")

if __name__ == "__main__":
    archive_current_results()
