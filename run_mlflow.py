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
    
    # Automatically sweep and destroy any Google Drive desktop.ini files
    def desktop_ini_scrubber():
        import time
        while True:
            if os.path.exists(local_mlruns):
                for root, dirs, files in os.walk(local_mlruns):
                    for file in files:
                        if file.lower() == "desktop.ini":
                            try:
                                os.remove(os.path.join(root, file))
                            except:
                                pass
                        elif file.lower() == "meta.yaml" and root != local_mlruns:
                            try:
                                meta_path = os.path.join(root, file)
                                with open(meta_path, 'r', encoding='utf-8') as f_meta:
                                    lines = f_meta.read().splitlines()
                                run_id = None
                                has_run_uuid = False
                                for line in lines:
                                    if line.startswith("run_id:"):
                                        run_id = line.split(":", 1)[1].strip().strip("'").strip('"')
                                    if line.startswith("run_uuid:"):
                                        has_run_uuid = True
                                if run_id and not has_run_uuid:
                                    with open(meta_path, 'a', encoding='utf-8') as f_meta:
                                        f_meta.write(f"\nrun_uuid: '{run_id}'\n")
                            except:
                                pass
            time.sleep(5)
            
    import threading
    scrubber_thread = threading.Thread(target=desktop_ini_scrubber, daemon=True)
    scrubber_thread.start()
                        
    sys.argv = ["mlflow", "ui", "--backend-store-uri", f"file:///{local_mlruns.replace(os.sep, '/')}"]
        
    cli()
