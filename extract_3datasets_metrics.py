import os
import pandas as pd
import mlflow

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
local_mlruns = os.path.abspath("mlruns").replace("\\", "/")
mlflow.set_tracking_uri(f"file:///{local_mlruns}")

experiments = mlflow.search_experiments()

data = []

for exp in experiments:
    runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
    if runs.empty:
        continue
    for idx, r in runs.iterrows():
        run_name = r.get("tags.mlflow.runName", r.get("run_id"))
        exp_name = exp.name
        
        # metrics
        m_dict = {c.replace("metrics.", ""): r[c] for c in runs.columns if c.startswith("metrics.") and pd.notna(r[c])}
        p_dict = {c.replace("params.", ""): r[c] for c in runs.columns if c.startswith("params.") and pd.notna(r[c])}
        
        m_dict.update({
            'experiment': exp_name,
            'run_name': run_name,
            'model_type': p_dict.get('model_type', p_dict.get('model', 'N/A')),
            'leak_free': p_dict.get('leak_free', 'N/A')
        })
        data.append(m_dict)

df = pd.DataFrame(data)
print("Total logged runs:", len(df))
print("Experiments:", df['experiment'].unique())

# Filter for the main full suite runs or representative runs per dataset and model
datasets = ['Altamira', 'Brumadinho', 'Mariana']
models = ['ZScore', 'IsolationForest', 'OneClassSVM', 'LSTM']

print("\n--- Summary of runs per dataset/experiment ---")
for d in datasets:
    print(f"\n=================== DATASET: {d} ===================")
    sub = df[df['experiment'].str.contains(d, case=False, na=False) | df['run_name'].str.contains(d, case=False, na=False)]
    for idx, row in sub.iterrows():
        print(f"Exp: {row['experiment']} | Run: {row['run_name']}")
        for col in sub.columns:
            if col not in ['experiment', 'run_name'] and pd.notna(row[col]):
                print(f"   {col}: {row[col]}")
