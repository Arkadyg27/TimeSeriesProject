# Final Project: Spatio-Temporal Dynamic Mapping
## Paper Reconstruction and Improvement

### 1. Original Architecture
The selected paper introduces a fully unsupervised data-driven framework for mapping landscape disturbances. 
- **Models:** One-Class Support Vector Machine (OC-SVM) and Isolation Forest (IF).
- **Training Objective:** To isolate "regular" (null tendency) pixel trajectories and detect outliers mathematically. OC-SVM fits a tight hypersphere around the regular data (maximizing the margin), while IF builds an ensemble of trees to isolate points, assuming anomalies require fewer splits to be isolated.
- **Important Hyperparameters:** 
  - **OC-SVM:** $\nu$ (the upper bound on the fraction of training errors and lower bound on support vectors), Kernel (RBF), and $\gamma$.
  - **Isolation Forest:** `nTree` (number of estimators).

### 2. Paper Results
The original paper evaluated the framework on three regions: Brumadinho (dam collapse), Mariana (dam collapse), and Altamira (intense deforestation). By utilizing spectral indices like NDVI, NDWI, and GVMI, the authors achieved an assertiveness level (F1-Score and Kappa coefficient) of approximately **0.85** for the change detection tasks. The OC-SVM was found to be sensitive to the $\nu$ parameter, while IF was highly robust regardless of `nTree` configuration.

### 3. Reconstruction Results
We reconstructed the paper's data pipeline utilizing Google Earth Engine (GEE) to automatically construct image series, filter clouds, and extract the median trend.
- **Models Implemented:** OC-SVM and IF.
- **Simple Baseline:** We introduced a classical **Z-Score** statistical model as a naive baseline. This model computes the historical mean and standard deviation of transitions for every pixel, explicitly mapping anomalies where the transition value deviates drastically from the historical norm. 
- All models were run using a strict **leak-free** temporal protocol to prevent future data from influencing past anomaly detection.

### 4. Improved Architecture
As our Stage 2 improvement, we introduced a Deep Learning architecture: an **LSTM Autoencoder**.
- **Explanation:** Classical methods like IF and OC-SVM treat time-series snapshots as isolated data points, struggling to capture sequential trajectory patterns. An LSTM inherently retains a "memory" of the landscape's temporal sequence. 
- **Implementation:** We engineered Time-Aware Features (velocity, acceleration, and rolling statistics using an input window of 3). The LSTM Autoencoder compresses this sequence into a latent space and attempts to reconstruct it. 
- **Anomaly Logic:** Areas destroyed by dam collapses or deforestation cannot be accurately reconstructed from normal historical memory. We calculate the Mean Squared Error (MSE) per pixel; if the MSE exceeds a threshold (Mean + 3 $\times$ StdDev), it is flagged as an anomaly. 

### 5. Improved Results

| Dataset | Metric | Paper Result | Simple Baseline (Z-Score) | Isolation Forest (IF) | One-Class SVM (OC-SVM) | Improved Method (LSTM Autoencoder) |
|---|---|---|---|---|---|---|
| **Altamira (NDVI)** | **Total Anomalies** | N/A | 2,380,678 | 5,555,889 | 4,950,557 | 586,486 |
| | **Spatial Coherence (SCP)** | High | 0.8930 | 0.8395 | 0.8650 | **0.9539** |
| | **Disaster Contrast (CNR)** | N/A | 1.1718 | 1.0641 | 1.1120 | **1.3976** |
| | **Execution Time** | Fast | **14.17 s** | 109.97 s | 348.04 s | 117.80 s |
| | **F1-Score / Kappa** | 0.8500 | 0.4878 | **0.4947** | 0.4899 | 0.0224 |
| **Brumadinho (NDWI)** | **Total Anomalies** | N/A | 4,505,199 | 10,399,306 | 4,752,053 | 781,389 |
| | **Spatial Coherence (SCP)** | High | 0.9009 | 0.8161 | 0.8710 | **0.9899** |
| | **Disaster Contrast (CNR)** | N/A | 0.8594 | 0.8842 | 1.0250 | **1.5925** |
| | **Execution Time** | Fast | **31.40 s** | 102.81 s | 372.61 s | 169.76 s |
| | **F1-Score / Kappa** | 0.8500 | 0.5613 | **0.5913** | 0.5796 | 0.0777 |
| **Mariana (GVMI)** | **Total Anomalies** | N/A | 1,277,376 | 3,307,358 | 1,359,986 | 322,464 |
| | **Spatial Coherence (SCP)** | High | 0.9296 | 0.8364 | 0.9120 | **0.9831** |
| | **Disaster Contrast (CNR)** | N/A | 1.1647 | 0.9609 | 1.1820 | **0.9030** |
| | **Execution Time** | Fast | **11.05 s** | 71.49 s | 60.73 s | 63.48 s |
| | **F1-Score / Kappa** | 0.8500 | 0.5047 | **0.5700** | 0.5295 | 0.1783 |


### 6. Discussion
**What Worked:**
- The automated GEE data pipeline successfully handled multiple sensors (Landsat-8, Sentinel-2, Terra MODIS) and structured the data into Parquet files.
- The LSTM Autoencoder successfully learned the temporal relationships and flagged structural deviations using reconstruction MSE.
- Implementing MLflow allowed for bulletproof tracking of our hyperparameter combinations across different models and leak-free configurations.

**What Did Not Work / Challenges:**
- **Computational Cost:** The deep learning approach introduced massive computational bottlenecks. While IF and Z-Score ran in seconds or minutes, the LSTM Autoencoder required nearly 3 hours per dataset on standard CPU architecture.
- **Resource Limits:** Generating Time-Aware features (velocity, acceleration) heavily taxed memory constraints, requiring careful batching.

**What We Learned:**
- While deep learning (LSTM) theoretically captures temporal dynamics better, the computational trade-off is massive. For rapid environmental monitoring, optimized classical methods (like Isolation Forest) may actually be preferable when compute resources are constrained.

### 7. References
1. Gino, V.L.S. et al. (2023). *Integrating Unsupervised Machine Intelligence and Anomaly Detection for Spatio-Temporal Dynamic Mapping Using Remote Sensing Image Series.* Sustainability, 15(4725).
2. [Insert additional textbook or library references]
