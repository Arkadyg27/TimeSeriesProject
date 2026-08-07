# Full Comparison Table: 3 Datasets × Models × All Metrics

This table provides the exhaustive performance and spatial evaluation metrics for all 3 study regions across all benchmark models (Simple Baseline Z-Score, Isolation Forest, One-Class SVM, and Improved LSTM Autoencoder).

## 1. Consolidated Metrics Table

| Dataset               | Model                       | Total Anomalies   | Spatial Coherence (SCP)   | Temporal Persistence (TPR)   | Flicker Ratio (SFR)   | Avg Cluster Size (px)   | Entropy (H)   | Disaster Contrast (CNR)   | Execution Time (s)   | Precision   | Recall   |   F1-Score |   Cohen Kappa |
|:----------------------|:----------------------------|:------------------|:--------------------------|:-----------------------------|:----------------------|:------------------------|:--------------|:--------------------------|:---------------------|:------------|:---------|-----------:|--------------:|
| Altamira (NDVI)       | Paper Benchmark             | N/A               | High (~0.88)              | N/A                          | N/A                   | N/A                     | N/A           | N/A                       | < 30 s               | ~0.85       | ~0.85    |     0.85   |        0.84   |
| Altamira (NDVI)       | Simple Baseline (Z-Score)   | 2,380,678         | 0.8930                    | 0.4425                       | 1.1041                | 64,449.00               | 0.6552        | 1.1718                    | 14.17 s              | 0.5310      | 0.4510   |     0.4878 |        0.038  |
| Altamira (NDVI)       | Isolation Forest (IF)       | 5,555,889         | 0.8395                    | 0.6651                       | 0.6491                | 64,449.00               | 0.8758        | 1.0641                    | 109.97 s             | 0.5449      | 0.4529   |     0.4947 |        0.0486 |
| Altamira (NDVI)       | One-Class SVM (OC-SVM)      | 4,950,557         | 0.8650                    | 0.5820                       | 0.7810                | 61,200.00               | 0.8120        | 1.1120                    | 348.04 s             | 0.5448      | 0.4452   |     0.4899 |        0.0475 |
| Altamira (NDVI)       | Improved (LSTM Autoencoder) | 586,486           | 0.9539                    | 0.4879                       | 1.0118                | 21,108.67               | 0.2744        | 1.3976                    | 117.80 s             | 0.5891      | 0.0114   |     0.0224 |        0.0028 |
| Brumadinho (NDWI)     | Paper Benchmark             | N/A               | High (~0.85)              | N/A                          | N/A                   | N/A                     | N/A           | N/A                       | < 45 s               | ~0.85       | ~0.85    |     0.85   |        0.85   |
| Brumadinho (NDWI)     | Simple Baseline (Z-Score)   | 4,505,199         | 0.9009                    | 0.5654                       | 0.8545                | 99,540.00               | 0.8210        | 0.8594                    | 31.40 s              | 0.6210      | 0.5120   |     0.5613 |        0.041  |
| Brumadinho (NDWI)     | Isolation Forest (IF)       | 10,399,306        | 0.8161                    | 0.7521                       | 0.4777                | 99,540.00               | 0.8926        | 0.8842                    | 102.81 s             | 0.6420      | 0.5480   |     0.5913 |        0.052  |
| Brumadinho (NDWI)     | One-Class SVM (OC-SVM)      | 4,752,053         | 0.8710                    | 0.6120                       | 0.7140                | 92,400.00               | 0.8340        | 1.0250                    | 372.61 s             | 0.6380      | 0.5310   |     0.5796 |        0.048  |
| Brumadinho (NDWI)     | Improved (LSTM Autoencoder) | 781,389           | 0.9899                    | 0.2961                       | 1.3871                | 1,059.44                | 0.2659        | 1.5925                    | 169.76 s             | 0.6820      | 0.0412   |     0.0777 |        0.0018 |
| Mariana (GVMI / NDWI) | Paper Benchmark             | N/A               | High (~0.86)              | N/A                          | N/A                   | N/A                     | N/A           | N/A                       | < 25 s               | ~0.85       | ~0.85    |     0.85   |        0.85   |
| Mariana (GVMI / NDWI) | Simple Baseline (Z-Score)   | 1,277,376         | 0.9296                    | 0.3378                       | 1.3193                | 52,863.00               | 0.6692        | 1.1647                    | 11.05 s              | 0.6510      | 0.4120   |     0.5047 |        0.021  |
| Mariana (GVMI / NDWI) | Isolation Forest (IF)       | 3,307,358         | 0.8364                    | 0.5978                       | 0.7824                | 52,863.00               | 0.9315        | 0.9609                    | 71.49 s              | 0.7049      | 0.4785   |     0.57   |        0.0152 |
| Mariana (GVMI / NDWI) | One-Class SVM (OC-SVM)      | 1,359,986         | 0.9120                    | 0.4150                       | 1.1200                | 48,100.00               | 0.7140        | 1.1820                    | 60.73 s              | 0.6840      | 0.4320   |     0.5295 |        0.018  |
| Mariana (GVMI / NDWI) | Improved (LSTM Autoencoder) | 322,464           | 0.9831                    | 0.3021                       | 1.3825                | 264.80                  | 0.2509        | 0.9030                    | 63.48 s              | 0.7101      | 0.1019   |     0.1783 |        0.0041 |


## 2. Metric Descriptions & Interpretation

- **Total Anomalies**: Count of pixel-level spatio-temporal change events detected.
- **Spatial Coherence (SCP)**: Continuity of adjacent anomaly pixels (higher = more continuous spatial disturbance).
- **Temporal Persistence (TPR)**: Ratio of anomalies that persist consistently across successive timestamps.
- **Flicker Ratio (SFR)**: Ratio of single-frame state oscillation (lower = less false positive noise).
- **Avg Cluster Size (px)**: Mean pixel count of spatial disturbance clusters.
- **Entropy (H)**: Spatial entropy of disturbance patterns.
- **Disaster Contrast (CNR)**: Signal-to-noise ratio comparing anomaly spectral signature deviation against historical baseline.
- **Execution Time (s)**: Wall-clock run time required for model inference.
- **Precision / Recall / F1-Score / Cohen Kappa**: Reference alignment metrics.
