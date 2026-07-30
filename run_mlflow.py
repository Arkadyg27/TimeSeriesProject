import mimetypes
import os
import sys
from mlflow.cli import cli

if __name__ == "__main__":
    # Fix the Windows bug where .js files are wrongly served as 'text/plain'
    mimetypes.add_type("application/javascript", ".js")
    
    # Opt-in to the file store backend for MLflow
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    
    # Reverting back to default file store because Colab scripts accidentally logged there
    sys.argv = ["mlflow", "ui"]
    cli()
