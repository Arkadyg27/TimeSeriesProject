import mimetypes
import os
import sys
from mlflow.cli import cli

if __name__ == "__main__":
    # Fix the Windows bug where .js files are wrongly served as 'text/plain'
    mimetypes.add_type("application/javascript", ".js")
    
    # Opt-in to the file store backend for MLflow
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    
    # Aggressively clear any hidden tracking URIs that might be causing SQLite to override our file store
    if "MLFLOW_TRACKING_URI" in os.environ:
        del os.environ["MLFLOW_TRACKING_URI"]

    # Force the local file store to avoid any lingering SQLite configs or Google Drive locking errors
    local_mlruns = os.path.join(os.getcwd(), "mlruns")
    
    # Automatically sweep and destroy any Google Drive desktop.ini files before booting
    if os.path.exists(local_mlruns):
        for root, dirs, files in os.walk(local_mlruns):
            for file in files:
                if file.lower() == "desktop.ini":
                    try:
                        os.remove(os.path.join(root, file))
                    except:
                        pass
                        
    sys.argv = ["mlflow", "ui", "--backend-store-uri", f"file:///{local_mlruns.replace(os.sep, '/')}"]
        
    cli()
