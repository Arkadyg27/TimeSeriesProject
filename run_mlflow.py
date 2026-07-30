import mimetypes
import os
import sys
from mlflow.cli import cli

if __name__ == "__main__":
    # Fix the Windows bug where .js files are wrongly served as 'text/plain'
    mimetypes.add_type("application/javascript", ".js")
    
    # Opt-in to the file store backend for MLflow
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    
    # Look for the live Google Drive database first to avoid sync delays
    gdrive_db = r"G:\My Drive\Study\MSc_CE_BGU\Time Series Analysis\Final Project\TimeSeriesProject\mlflow.db"
    if os.path.exists(gdrive_db):
        db_uri = f"sqlite:///{gdrive_db.replace(os.sep, '/')}"
    else:
        db_uri = "sqlite:///mlflow.db"

    # Simulate the 'mlflow ui' command
    sys.argv = ["mlflow", "ui", "--backend-store-uri", db_uri]
    cli()
