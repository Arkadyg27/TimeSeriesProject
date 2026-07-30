import mimetypes
import os
import sys
from mlflow.cli import cli

if __name__ == "__main__":
    # Fix the Windows bug where .js files are wrongly served as 'text/plain'
    mimetypes.add_type("application/javascript", ".js")
    
    # Opt-in to the file store backend for MLflow
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    
    # Simulate the 'mlflow ui' command
    sys.argv = ["mlflow", "ui", "--backend-store-uri", "sqlite:///mlflow.db"]
    cli()
