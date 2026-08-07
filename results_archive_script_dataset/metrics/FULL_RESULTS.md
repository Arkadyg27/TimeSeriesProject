# Full Experimental Results & Metrics Summary

This document contains the complete quantitative results for the **Spatio-Temporal Dynamic Mapping** project across all benchmark datasets, models, temporal leakage protocols, and MLflow experiment runs.

## 1. Executive Summary Table (Paper Comparison & Model Breakdown)

| Dataset | Spectral Index | Simple Baseline (Z-Score) Anomalies | Isolation Forest (IF) Anomalies | One-Class SVM (OC-SVM) Anomalies | Improved Method (LSTM Autoencoder) Anomalies | Paper Reported Benchmark |
|---|---|---|---|---|---|---|
| **Altamira** | NDVI | 2,380,678 | 5,555,889 | 4,950,557 | 586,486 | F1 / Kappa ~ 0.85 |
| **Brumadinho** | NDWI | 4,505,199 | 10,399,306 | 4,752,053 | 781,389 | F1 / Kappa ~ 0.85 |
| **Mariana** | GVMI / NDWI | 1,277,376 | 3,307,358 | 1,359,986 | 322,464 | F1 / Kappa ~ 0.85 |

## 2. GeoTIFF Anomaly Rasters Spatial Metrics

Unsupervised spatial metrics computed directly from output GeoTIFF rasters:

- **SCP (Spatial Coherence Index)**: Degree of spatial continuity among adjacent pixels (closer to 1.0 = higher spatial clustering).
- **Avg Cluster Size**: Average size (in pixels) of contiguous anomaly clusters.
- **Entropy (H)**: Spatial disturbance entropy metric.

| dataset    | model           | protocol   | file_name                                          |   total_pixels |   total_anomalies |   anomaly_ratio |   spatial_coherence_scp |   avg_cluster_size |   entropy_h |
|:-----------|:----------------|:-----------|:---------------------------------------------------|---------------:|------------------:|----------------:|------------------------:|-------------------:|------------:|
| Altamira   | IsolationForest | leaky      | Altamira_NDVI_IsolationForest_leaky_est_100.tif    |          64449 |                 0 |               0 |                       0 |                  0 |           0 |
| Altamira   | IsolationForest | leaky      | Altamira_NDVI_IsolationForest_leaky_est_20.tif     |          64449 |                 0 |               0 |                       0 |                  0 |           0 |
| Altamira   | IsolationForest | leaky      | Altamira_NDVI_IsolationForest_leaky_est_40.tif     |          64449 |                 0 |               0 |                       0 |                  0 |           0 |
| Altamira   | IsolationForest | leaky      | Altamira_NDVI_IsolationForest_leaky_est_60.tif     |          64449 |                 0 |               0 |                       0 |                  0 |           0 |
| Altamira   | IsolationForest | leaky      | Altamira_NDVI_IsolationForest_leaky_est_80.tif     |          64449 |                 0 |               0 |                       0 |                  0 |           0 |
| Brumadinho | IsolationForest | leaky      | Brumadinho_NDWI_IsolationForest_leaky_est_100.tif  |          99540 |                 0 |               0 |                       0 |                  0 |           0 |
| Brumadinho | IsolationForest | leaky      | Brumadinho_NDWI_IsolationForest_leaky_est_20.tif   |          99540 |                 0 |               0 |                       0 |                  0 |           0 |
| Brumadinho | IsolationForest | leaky      | Brumadinho_NDWI_IsolationForest_leaky_est_40.tif   |          99540 |                 0 |               0 |                       0 |                  0 |           0 |
| Brumadinho | IsolationForest | leaky      | Brumadinho_NDWI_IsolationForest_leaky_est_60.tif   |          99540 |                 0 |               0 |                       0 |                  0 |           0 |
| Brumadinho | IsolationForest | leaky      | Brumadinho_NDWI_IsolationForest_leaky_est_80.tif   |          99540 |                 0 |               0 |                       0 |                  0 |           0 |
| Mariana    | IsolationForest | leaky      | Mariana_GVMI_IsolationForest_leaky_est_100.tif     |          52863 |                 0 |               0 |                       0 |                  0 |           0 |
| Mariana    | IsolationForest | leaky      | Mariana_GVMI_IsolationForest_leaky_est_20.tif      |          52863 |                 0 |               0 |                       0 |                  0 |           0 |
| Mariana    | IsolationForest | leaky      | Mariana_GVMI_IsolationForest_leaky_est_40.tif      |          52863 |                 0 |               0 |                       0 |                  0 |           0 |
| Mariana    | IsolationForest | leaky      | Mariana_GVMI_IsolationForest_leaky_est_60.tif      |          52863 |                 0 |               0 |                       0 |                  0 |           0 |
| Mariana    | IsolationForest | leaky      | Mariana_GVMI_IsolationForest_leaky_est_80.tif      |          52863 |                 0 |               0 |                       0 |                  0 |           0 |
| Altamira   | OneClassSVM     | leaky      | Altamira_NDVI_OneClassSVM_leaky_nu_0.025.tif       |          64449 |                 0 |               0 |                       0 |                  0 |           0 |
| Altamira   | IsolationForest | leak_free  | Altamira_NDVI_IsolationForest_leakfree_est_100.tif |          64449 |                 0 |               0 |                       0 |                  0 |           0 |
| Altamira   | IsolationForest | leak_free  | Altamira_NDVI_IsolationForest_leakfree_est_20.tif  |          64449 |                 0 |               0 |                       0 |                  0 |           0 |
| Altamira   | IsolationForest | leak_free  | Altamira_NDVI_IsolationForest_leakfree_est_40.tif  |          64449 |                 0 |               0 |                       0 |                  0 |           0 |
| Altamira   | IsolationForest | leak_free  | Altamira_NDVI_IsolationForest_leakfree_est_60.tif  |          64449 |                 0 |               0 |                       0 |                  0 |           0 |
| Altamira   | IsolationForest | leak_free  | Altamira_NDVI_IsolationForest_leakfree_est_80.tif  |          64449 |                 0 |               0 |                       0 |                  0 |           0 |
| Mariana    | IsolationForest | leak_free  | Mariana_GVMI_IsolationForest_leakfree_est_20.tif   |          52863 |                 0 |               0 |                       0 |                  0 |           0 |
| Mariana    | IsolationForest | leak_free  | Mariana_GVMI_IsolationForest_leakfree_est_40.tif   |          52863 |                 0 |               0 |                       0 |                  0 |           0 |
| Altamira   | OneClassSVM     | V3/Deep    | Altamira_NDVI_OCSVM_0.025.tif                      |          64449 |                 0 |               0 |                       0 |                  0 |           0 |
| Altamira   | OneClassSVM     | V3/Deep    | Alt_OCSVM_modis_0.025.tif                          |          64449 |                 0 |               0 |                       0 |                  0 |           0 |
| Unknown    | IsolationForest | V3/Deep    | test.tif                                           |            100 |                 0 |               0 |                       0 |                  0 |           0 |


## 3. MLflow Experiments Summary

### Experiment: `DeepLearning_Altamira_NDVI_LSTM_AE` (Total Runs: 21)

| runName    | model           |   epochs |   execution_time_seconds |   cohen_kappa |    f1_score |   anomaly_threshold |   final_train_loss |
|:-----------|:----------------|---------:|-------------------------:|--------------:|------------:|--------------------:|-------------------:|
| Ep20_Pct99 | LSTMAutoencoder |       20 |                 140.006  |  nan          | nan         |           0.282091  |          0.0222244 |
| Ep20_Pct95 | LSTMAutoencoder |       20 |                 123.892  |  nan          | nan         |           0.115573  |          0.0222244 |
| Ep20_Pct90 | LSTMAutoencoder |       20 |                 107.594  |  nan          | nan         |           0.0590184 |          0.0222244 |
| Ep10_Pct99 | LSTMAutoencoder |       10 |                  61.3262 |  nan          | nan         |           0.282828  |          0.0223725 |
| Ep10_Pct95 | LSTMAutoencoder |       10 |                  46.0554 |  nan          | nan         |           0.116138  |          0.0223725 |
| Ep10_Pct90 | LSTMAutoencoder |       10 |                  30.7007 |  nan          | nan         |           0.0595945 |          0.0223725 |
| Ep20_Pct99 | LSTMAutoencoder |       20 |                6006.66   |  nan          | nan         |           0.281915  |          0.0222964 |
| Ep20_Pct95 | LSTMAutoencoder |       20 |                5986.29   |  nan          | nan         |           0.115272  |          0.0222964 |
| Ep20_Pct90 | LSTMAutoencoder |       20 |                5964.91   |  nan          | nan         |           0.0589497 |          0.0222964 |
| Ep10_Pct99 | LSTMAutoencoder |       10 |                3013.82   |    0.00281444 |   0.0223797 |           0.282452  |          0.0223645 |


### Experiment: `DeepLearning_Brumadinho_NDVI_LSTM_AE` (Total Runs: 18)

| runName    | model           |   epochs |   execution_time_seconds |   anomaly_threshold |   final_train_loss |
|:-----------|:----------------|---------:|-------------------------:|--------------------:|-------------------:|
| Ep20_Pct99 | LSTMAutoencoder |       20 |                 196.913  |           0.287515  |          0.0227795 |
| Ep20_Pct95 | LSTMAutoencoder |       20 |                 174.781  |           0.117857  |          0.0227795 |
| Ep20_Pct90 | LSTMAutoencoder |       20 |                 153.443  |           0.0537363 |          0.0227795 |
| Ep10_Pct99 | LSTMAutoencoder |       10 |                  88.6074 |           0.287438  |          0.0231786 |
| Ep10_Pct95 | LSTMAutoencoder |       10 |                  65.91   |           0.118484  |          0.0231786 |
| Ep10_Pct90 | LSTMAutoencoder |       10 |                  44.5244 |           0.055673  |          0.0231786 |
| Ep20_Pct99 | LSTMAutoencoder |       20 |                7731.1    |           0.286708  |          0.0228175 |
| Ep20_Pct95 | LSTMAutoencoder |       20 |                7703.84   |           0.117722  |          0.0228175 |
| Ep20_Pct90 | LSTMAutoencoder |       20 |                7675.73   |           0.0539896 |          0.0228175 |
| Ep10_Pct99 | LSTMAutoencoder |       10 |                3870.82   |           0.287642  |          0.0240417 |


### Experiment: `DeepLearning_Brumadinho_NDWI_LSTM_AE` (Total Runs: 12)

| runName    | model           |   epochs |   execution_time_seconds |   anomaly_threshold |   final_train_loss |
|:-----------|:----------------|---------:|-------------------------:|--------------------:|-------------------:|
| Ep20_Pct99 | LSTMAutoencoder |       20 |                 201.635  |           0.174547  |          0.0148812 |
| Ep20_Pct95 | LSTMAutoencoder |       20 |                 178.981  |           0.0773625 |          0.0148812 |
| Ep20_Pct90 | LSTMAutoencoder |       20 |                 154.587  |           0.0402107 |          0.0148812 |
| Ep10_Pct99 | LSTMAutoencoder |       10 |                  88.8733 |           0.175509  |          0.0152978 |
| Ep10_Pct95 | LSTMAutoencoder |       10 |                  66.866  |           0.0785784 |          0.0152978 |
| Ep10_Pct90 | LSTMAutoencoder |       10 |                  44.5721 |           0.0415861 |          0.0152978 |
| Ep20_Pct99 | LSTMAutoencoder |       20 |                 192.906  |           0.175027  |          0.015072  |
| Ep20_Pct95 | LSTMAutoencoder |       20 |                 172.376  |           0.0776323 |          0.015072  |
| Ep20_Pct90 | LSTMAutoencoder |       20 |                 150.203  |           0.0405478 |          0.015072  |
| Ep10_Pct99 | LSTMAutoencoder |       10 |                  86.5438 |           0.176991  |          0.0158911 |


### Experiment: `DeepLearning_Mariana_GVMI_LSTM_AE` (Total Runs: 12)

| runName    | model           |   epochs |   execution_time_seconds |   cohen_kappa |   f1_score |   anomaly_threshold |   final_train_loss |
|:-----------|:----------------|---------:|-------------------------:|--------------:|-----------:|--------------------:|-------------------:|
| Ep20_Pct99 | LSTMAutoencoder |       20 |                 106.661  |  -0.00125365  |  0.0184881 |           0.248644  |          0.026531  |
| Ep20_Pct95 | LSTMAutoencoder |       20 |                  92.1494 |   0.000111194 |  0.0934051 |           0.135451  |          0.026531  |
| Ep20_Pct90 | LSTMAutoencoder |       20 |                  78.8511 |   0.00406731  |  0.178253  |           0.0795082 |          0.026531  |
| Ep10_Pct99 | LSTMAutoencoder |       10 |                  47.1949 |  -0.00314802  |  0.0166311 |           0.248696  |          0.0265345 |
| Ep10_Pct95 | LSTMAutoencoder |       10 |                  33.5745 |  -0.00322971  |  0.0903759 |           0.135551  |          0.0265345 |
| Ep10_Pct90 | LSTMAutoencoder |       10 |                  20.0588 |   0.000419011 |  0.175243  |           0.0796633 |          0.0265345 |
| Ep20_Pct99 | LSTMAutoencoder |       20 |                  99.4342 |   0.000927646 |  0.0206264 |           0.248897  |          0.0264972 |
| Ep20_Pct95 | LSTMAutoencoder |       20 |                  87.3542 |  -0.00238738  |  0.0911396 |           0.135324  |          0.0264972 |
| Ep20_Pct90 | LSTMAutoencoder |       20 |                  75.2222 |   0.00126814  |  0.175944  |           0.0794801 |          0.0264972 |
| Ep10_Pct99 | LSTMAutoencoder |       10 |                  43.6563 |  -0.00317667  |  0.016603  |           0.248853  |          0.0265262 |


### Experiment: `DeepLearning_Mariana_NDWI_LSTM_AE` (Total Runs: 12)

| runName    | model           |   epochs |   execution_time_seconds |   anomaly_threshold |   final_train_loss |
|:-----------|:----------------|---------:|-------------------------:|--------------------:|-------------------:|
| Ep20_Pct99 | LSTMAutoencoder |       20 |                 105.754  |           0.228623  |          0.0279346 |
| Ep20_Pct95 | LSTMAutoencoder |       20 |                  92.2238 |           0.1304    |          0.0279346 |
| Ep20_Pct90 | LSTMAutoencoder |       20 |                  78.621  |           0.0837939 |          0.0279346 |
| Ep10_Pct99 | LSTMAutoencoder |       10 |                  46.8073 |           0.225918  |          0.0279604 |
| Ep10_Pct95 | LSTMAutoencoder |       10 |                  33.5194 |           0.130162  |          0.0279604 |
| Ep10_Pct90 | LSTMAutoencoder |       10 |                  19.4762 |           0.084013  |          0.0279604 |
| Ep20_Pct99 | LSTMAutoencoder |       20 |                  98.1922 |           0.22566   |          0.0278554 |
| Ep20_Pct95 | LSTMAutoencoder |       20 |                  87.5021 |           0.129481  |          0.0278554 |
| Ep20_Pct90 | LSTMAutoencoder |       20 |                  75.2169 |           0.0838118 |          0.0278554 |
| Ep10_Pct99 | LSTMAutoencoder |       10 |                  44.0536 |           0.229586  |          0.028053  |


### Experiment: `DynaLand_Altamira_NDVI` (Total Runs: 36)

| runName                         |   n_estimators | model_type      | leak_free   |    nu |   execution_time_seconds |   cohen_kappa |   f1_score |
|:--------------------------------|---------------:|:----------------|:------------|------:|-------------------------:|--------------:|-----------:|
| OneClassSVM_leaky_nu_0.025      |                | OneClassSVM     | False       | 0.025 |                 435.018  |     0.0079317 |   0.474295 |
| OneClassSVM_leakfree_nu_0.1     |                | OneClassSVM     | True        | 0.1   |                1387.47   |     0.0434661 |   0.515407 |
| OneClassSVM_leakfree_nu_0.025   |                | OneClassSVM     | True        | 0.025 |                 348.043  |     0.0474987 |   0.48995  |
| OneClassSVM_leaky_nu_0.1        |                | OneClassSVM     | False       | 0.1   |                1633.88   |     0.011557  |   0.500672 |
| IsolationForest_leaky_est_80    |             80 | IsolationForest | False       |       |                  71.3531 |     0.048589  |   0.494683 |
| IsolationForest_leaky_est_60    |             60 | IsolationForest | False       |       |                  54.8851 |     0.0489922 |   0.494803 |
| IsolationForest_leaky_est_40    |             40 | IsolationForest | False       |       |                  36.0294 |     0.0500819 |   0.495395 |
| IsolationForest_leaky_est_20    |             20 | IsolationForest | False       |       |                  23.8514 |     0.058426  |   0.49641  |
| IsolationForest_leakfree_est_80 |             80 | IsolationForest | True        |       |                 145.016  |     0.032427  |   0.505719 |
| IsolationForest_leakfree_est_60 |             60 | IsolationForest | True        |       |                 127.105  |     0.0346522 |   0.50782  |


### Experiment: `DynaLand_Brumadinho_NDVI` (Total Runs: 30)

| runName                       | n_estimators   | model_type   | leak_free   |    nu |   execution_time_seconds |
|:------------------------------|:---------------|:-------------|:------------|------:|-------------------------:|
| OneClassSVM_leakfree_nu_0.1   |                | OneClassSVM  | True        | 0.1   |                 3269.18  |
| OneClassSVM_leakfree_nu_0.05  |                | OneClassSVM  | True        | 0.05  |                 1675.89  |
| OneClassSVM_leakfree_nu_0.01  |                | OneClassSVM  | True        | 0.01  |                  398.73  |
| OneClassSVM_leakfree_nu_0.005 |                | OneClassSVM  | True        | 0.005 |                  240.555 |
| OneClassSVM_leakfree_nu_0.001 |                | OneClassSVM  | True        | 0.001 |                  110.844 |
| OneClassSVM_leaky_nu_0.1      |                | OneClassSVM  | False       | 0.1   |                10226.1   |
| OneClassSVM_leaky_nu_0.05     |                | OneClassSVM  | False       | 0.05  |                 5087.64  |
| OneClassSVM_leaky_nu_0.01     |                | OneClassSVM  | False       | 0.01  |                 1020.46  |
| OneClassSVM_leaky_nu_0.005    |                | OneClassSVM  | False       | 0.005 |                  514.51  |
| OneClassSVM_leaky_nu_0.001    |                | OneClassSVM  | False       | 0.001 |                  106.47  |


### Experiment: `DynaLand_Brumadinho_NDWI` (Total Runs: 39)

| runName                       | n_estimators   | model_type   | leak_free   |    nu |   execution_time_seconds |
|:------------------------------|:---------------|:-------------|:------------|------:|-------------------------:|
| OneClassSVM_leakfree_nu_0.1   |                | OneClassSVM  | True        | 0.1   |                3351.68   |
| OneClassSVM_leakfree_nu_0.05  |                | OneClassSVM  | True        | 0.05  |                1661.3    |
| OneClassSVM_leakfree_nu_0.01  |                | OneClassSVM  | True        | 0.01  |                 372.607  |
| OneClassSVM_leakfree_nu_0.005 |                | OneClassSVM  | True        | 0.005 |                 225.255  |
| OneClassSVM_leakfree_nu_0.001 |                | OneClassSVM  | True        | 0.001 |                 106.256  |
| OneClassSVM_leaky_nu_0.1      |                | OneClassSVM  | False       | 0.1   |               10028.8    |
| OneClassSVM_leaky_nu_0.05     |                | OneClassSVM  | False       | 0.05  |                5089.22   |
| OneClassSVM_leaky_nu_0.01     |                | OneClassSVM  | False       | 0.01  |                 806.154  |
| OneClassSVM_leaky_nu_0.005    |                | OneClassSVM  | False       | 0.005 |                 400.969  |
| OneClassSVM_leaky_nu_0.001    |                | OneClassSVM  | False       | 0.001 |                  84.7523 |


### Experiment: `DynaLand_Mariana_GVMI` (Total Runs: 30)

| runName                       | n_estimators   | model_type   | leak_free   |    nu |   execution_time_seconds |   cohen_kappa |   f1_score |
|:------------------------------|:---------------|:-------------|:------------|------:|-------------------------:|--------------:|-----------:|
| OneClassSVM_leakfree_nu_0.1   |                | OneClassSVM  | True        | 0.1   |                1342.55   |           nan |        nan |
| OneClassSVM_leakfree_nu_0.05  |                | OneClassSVM  | True        | 0.05  |                 511.308  |           nan |        nan |
| OneClassSVM_leakfree_nu_0.01  |                | OneClassSVM  | True        | 0.01  |                 108.439  |           nan |        nan |
| OneClassSVM_leakfree_nu_0.005 |                | OneClassSVM  | True        | 0.005 |                  60.7292 |           nan |        nan |
| OneClassSVM_leakfree_nu_0.001 |                | OneClassSVM  | True        | 0.001 |                  25.8043 |           nan |        nan |
| OneClassSVM_leaky_nu_0.1      |                | OneClassSVM  | False       | 0.1   |                1359.06   |           nan |        nan |
| OneClassSVM_leaky_nu_0.05     |                | OneClassSVM  | False       | 0.05  |                 682.11   |           nan |        nan |
| OneClassSVM_leaky_nu_0.01     |                | OneClassSVM  | False       | 0.01  |                 137.47   |           nan |        nan |
| OneClassSVM_leaky_nu_0.005    |                | OneClassSVM  | False       | 0.005 |                  70.6353 |           nan |        nan |
| OneClassSVM_leaky_nu_0.001    |                | OneClassSVM  | False       | 0.001 |                  18.5522 |           nan |        nan |


### Experiment: `DynaLand_Mariana_NDWI` (Total Runs: 30)

| runName                       | n_estimators   | model_type   | leak_free   |    nu |   execution_time_seconds |
|:------------------------------|:---------------|:-------------|:------------|------:|-------------------------:|
| OneClassSVM_leakfree_nu_0.1   |                | OneClassSVM  | True        | 0.1   |                1912.33   |
| OneClassSVM_leakfree_nu_0.05  |                | OneClassSVM  | True        | 0.05  |                 922.768  |
| OneClassSVM_leakfree_nu_0.01  |                | OneClassSVM  | True        | 0.01  |                 182.072  |
| OneClassSVM_leakfree_nu_0.005 |                | OneClassSVM  | True        | 0.005 |                 104.917  |
| OneClassSVM_leakfree_nu_0.001 |                | OneClassSVM  | True        | 0.001 |                  43.5106 |
| OneClassSVM_leaky_nu_0.1      |                | OneClassSVM  | False       | 0.1   |                8936.64   |
| OneClassSVM_leaky_nu_0.05     |                | OneClassSVM  | False       | 0.05  |                4860.37   |
| OneClassSVM_leaky_nu_0.01     |                | OneClassSVM  | False       | 0.01  |                1191.11   |
| OneClassSVM_leaky_nu_0.005    |                | OneClassSVM  | False       | 0.005 |                 445.773  |
| OneClassSVM_leaky_nu_0.001    |                | OneClassSVM  | False       | 0.001 |                  90.7377 |


### Experiment: `Simple_Baseline_ZScore` (Total Runs: 10)

| runName                           | model_type      | leak_free   |   execution_time_seconds |
|:----------------------------------|:----------------|:------------|-------------------------:|
| Mariana_NDWI_Baseline_leakfree    | Baseline_ZScore | True        |                 5.91489  |
| Mariana_NDWI_Baseline_leaky       | Baseline_ZScore | False       |                 0.286364 |
| Mariana_GVMI_Baseline_leakfree    | Baseline_ZScore | True        |                 5.02803  |
| Mariana_GVMI_Baseline_leaky       | Baseline_ZScore | False       |                 0.352165 |
| Brumadinho_NDWI_Baseline_leakfree | Baseline_ZScore | True        |                17.6196   |
| Brumadinho_NDWI_Baseline_leaky    | Baseline_ZScore | False       |                 0.666966 |
| Brumadinho_NDVI_Baseline_leakfree | Baseline_ZScore | True        |                17.5455   |
| Brumadinho_NDVI_Baseline_leaky    | Baseline_ZScore | False       |                 0.694311 |
| Altamira_NDVI_Baseline_leakfree   | Baseline_ZScore | True        |                16.1682   |
| Altamira_NDVI_Baseline_leaky      | Baseline_ZScore | False       |                 0.660329 |


### Experiment: `Unsupervised_Full_Suite_Altamira` (Total Runs: 6)

| runName                   | model            |   n_estimators |   execution_time_seconds |   spatial_coherence_scp |   temporal_entropy_h |
|:--------------------------|:-----------------|---------------:|-------------------------:|------------------------:|---------------------:|
| Altamira_LSTM_Autoencoder | LSTM Autoencoder |                |                 117.797  |                0.953925 |             0.274418 |
| Altamira_Isolation_Forest | Isolation Forest |             40 |                 109.971  |                0.839523 |             0.875756 |
| Altamira_ZScore_Baseline  | Z-Score Baseline |                |                  14.1743 |                0.893047 |             0.65516  |
| Altamira_LSTM_Autoencoder | LSTM Autoencoder |                |                  34.4388 |                0.957049 |             0.275067 |
| Altamira_Isolation_Forest | Isolation Forest |             40 |                  90.0308 |                0.839523 |             0.875756 |
| Altamira_ZScore_Baseline  | Z-Score Baseline |                |                  13.9079 |                0.893047 |             0.65516  |


### Experiment: `Unsupervised_Full_Suite_Brumadinho` (Total Runs: 6)

| runName                     | model            |   n_estimators |   execution_time_seconds |   spatial_coherence_scp |   temporal_entropy_h |
|:----------------------------|:-----------------|---------------:|-------------------------:|------------------------:|---------------------:|
| Brumadinho_LSTM_Autoencoder | LSTM Autoencoder |                |                 169.765  |                0.989948 |             0.265898 |
| Brumadinho_Isolation_Forest | Isolation Forest |             40 |                 102.813  |                0.816117 |             0.892606 |
| Brumadinho_ZScore_Baseline  | Z-Score Baseline |                |                  31.4046 |                0.900876 |             0.820954 |
| Brumadinho_LSTM_Autoencoder | LSTM Autoencoder |                |                  47.7332 |                0.98775  |             0.261864 |
| Brumadinho_Isolation_Forest | Isolation Forest |             40 |                 100.375  |                0.816117 |             0.892606 |
| Brumadinho_ZScore_Baseline  | Z-Score Baseline |                |                  15.4001 |                0.900876 |             0.820954 |


### Experiment: `Unsupervised_Full_Suite_Mariana` (Total Runs: 6)

| runName                  | model            |   n_estimators |   execution_time_seconds |   spatial_coherence_scp |   temporal_entropy_h |
|:-------------------------|:-----------------|---------------:|-------------------------:|------------------------:|---------------------:|
| Mariana_LSTM_Autoencoder | LSTM Autoencoder |                |                 63.4843  |                0.983083 |             0.250931 |
| Mariana_Isolation_Forest | Isolation Forest |             40 |                 71.4914  |                0.836406 |             0.931465 |
| Mariana_ZScore_Baseline  | Z-Score Baseline |                |                 11.0454  |                0.929616 |             0.669237 |
| Mariana_LSTM_Autoencoder | LSTM Autoencoder |                |                 18.3607  |                0.982543 |             0.26498  |
| Mariana_Isolation_Forest | Isolation Forest |             40 |                 41.9355  |                0.836406 |             0.931465 |
| Mariana_ZScore_Baseline  | Z-Score Baseline |                |                  4.88604 |                0.929616 |             0.669237 |

