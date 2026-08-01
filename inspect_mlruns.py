import os
import mlflow
import pandas as pd

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
local_mlruns = os.path.abspath("mlruns").replace("\\", "/")
mlflow.set_tracking_uri(f"file:///{local_mlruns}")

experiments = mlflow.search_experiments()

print(f"=== MLFLOW EXPERIMENTS SUMMARY ({len(experiments)} experiments) ===")
for exp in experiments:
    print(f"\nExperiment: {exp.name} (ID: {exp.experiment_id})")
    runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
    if runs.empty:
        print("  No runs found.")
        continue
    
    print(f"  Total runs: {len(runs)}")
    metric_cols = [c for c in runs.columns if c.startswith("metrics.")]
    param_cols = [c for c in runs.columns if c.startswith("params.")]
    
    # Print metrics present across these runs
    metrics_summary = {}
    for mc in metric_cols:
        mname = mc.replace("metrics.", "")
        vals = runs[mc].dropna()
        if len(vals) > 0:
            metrics_summary[mname] = (vals.min(), vals.max(), vals.mean())
            
    print("  Unsupervised Metrics Logged:")
    for mname, (vmin, vmax, vmean) in metrics_summary.items():
        print(f"    - {mname}: min={vmin:.4f}, max={vmax:.4f}, mean={vmean:.4f}")
    
    print("\n  Sample Runs Detail:")
    for idx, r in runs.head(5).iterrows():
        run_name = r.get("tags.mlflow.runName", r.get("run_id"))
        leak_free = r.get("params.leak_free", "N/A")
        model_type = r.get("params.model_type", "N/A")
        print(f"    Run: {run_name} | Model: {model_type} | leak_free={leak_free}")
        for mc in metric_cols:
            mname = mc.replace("metrics.", "")
            val = r[mc]
            if pd.notna(val):
                print(f"      {mname}: {val}")
