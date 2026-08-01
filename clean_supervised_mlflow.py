import os
import shutil
import mlflow

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
local_mlruns = os.path.abspath("mlruns").replace("\\", "/")
mlflow.set_tracking_uri(f"file:///{local_mlruns}")

experiments = mlflow.search_experiments()
removed = False

for exp in experiments:
    if exp.name == "Supervised_Evaluation_F1_Kappa":
        print(f"Removing MLflow experiment: {exp.name} (ID: {exp.experiment_id})...")
        exp_dir = os.path.join("mlruns", exp.experiment_id)
        if os.path.exists(exp_dir):
            shutil.rmtree(exp_dir, ignore_errors=True)
            print("Successfully deleted experiment directory!")
            removed = True

if not removed:
    print("No Supervised_Evaluation_F1_Kappa experiment found in mlruns.")
