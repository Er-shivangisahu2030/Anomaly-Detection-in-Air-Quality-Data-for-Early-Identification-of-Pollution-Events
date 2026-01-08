# Delhi Air Quality Anomaly Detection

A complete **time-series anomaly detection project** on **Delhi’s air quality** using both **classical machine learning (Isolation Forest)** and **deep learning (LSTM)** approaches.  
The project is built on a **merged dataset of AQI, fire activity, and weather variables** and compares anomaly patterns detected by tree-based and sequence-based models.

---

## 1. Project Overview

This project aims to detect **anomalous air-pollution days in Delhi** over multiple years by combining:

- Air quality and pollutant data
- Fire activity indicators
- Meteorological conditions

Two anomaly detection pipelines are implemented:

- **Isolation Forest** (unsupervised, tree-based)
- **LSTM (PyTorch)** with MinMax scaling for time-series modeling

The outputs from both models are analyzed and compared across seasons and pollution regimes.

---

## Key Goals

- Build a clean, unified **daily time-series dataset** from AQI, fire, and weather sources  
- Engineer **domain-specific features** such as rolling PM2.5 statistics and deviation indicators  
- Train and evaluate:
  - Isolation Forest for unsupervised anomaly detection
  - LSTM-based sequence model for anomaly scoring
- Compare anomaly patterns across **seasons and AQI categories**

---

## 2. Data Sources and Structure

The core dataset used for modeling is:
delhi_air_anomaly_ready.csv

It is created by merging three underlying datasets.

---

### 2.1 Source Datasets

#### AQI / Pollutant Dataset (Delhi)

- Daily city-level pollutant readings:
  - pm2.5, pm10, no2, so2, co, o3
- Computed AQI values and `aqibucket` classes (Good, Moderate, Poor, etc.)
- Calendar information:
  - date, year, month, season

#### Fire Count Dataset

- Daily aggregated fire hotspot counts: `firecount`
- `allindiapm25` as a regional PM2.5 proxy linked to fire activity

#### Weather Dataset (Delhi)

- Daily meteorological variables:
  - temperaturecelsius
  - humidity
  - windkph
- Season identifiers aligned with pollution dates

All datasets are merged on the **date** field to create a unified daily time series.

---

### 2.2 Engineered Anomaly Dataset

The merged and feature-engineered dataset is stored as:

delhi_air_anomaly_ready.csv

#### Main Columns

- **Time information**:
  - date, year, month, season
- **Pollutants**:
  - pm2.5, pm10, no2, so2, co, o3
- **Weather**:
  - temperaturecelsius, humidity, windkph
- **Fire / regional indicators**:
  - firecount, allindiapm25
- **Rolling statistics (PM2.5)**:
  - pm257dayavg – 7-day moving average
  - pm257daystd – 7-day moving standard deviation
- **Derived feature**:
  - pm25deviation – deviation from 7-day mean
- **AQI information**:
  - aqi, aqibucket

This file is the **starting point** for both anomaly detection pipelines.

---

## 3. Anomaly Detection Pipelines

Two independent anomaly detection outputs are generated and saved as separate CSV files.

---

### 3.1 Isolation Forest Pipeline

**Output file**
delhi_air_anomaly_with_iforest_train_test.csv

**Idea**  
An unsupervised Isolation Forest model is trained on pollution, weather, fire, and engineered features to identify days that are statistically unusual.

**Conceptual Steps** 

1. Load `delhi_air_anomaly_ready.csv'
2. Select features:
   - pm2.5, pm10, no2, so2, co, o3
   - temperaturecelsius, humidity, windkph
   - firecount, allindiapm25
   - pm257dayavg, pm257daystd, pm25deviation
3. Handle missing values (imputation or removal)
4. Maintain chronological order for train/test splits (if used)
5. Train Isolation Forest with tuned hyperparameters
6. Generate:
   - `iforestscore` – raw anomaly score
   - `iforestlabel` – binary anomaly flag
7. Save results to CSV

---

### 3.2 LSTM-Based Anomaly Pipeline (PyTorch)

**Output file**
delhi_air_anomaly_with_lstm_minmax_pytorch.csv

**Idea**  
An LSTM model learns normal temporal behavior of air quality (primarily PM2.5).  
High prediction or reconstruction error is treated as an anomaly.

**Conceptual Steps**

1. Load `delhi_air_anomaly_ready.csv`
2. Select target variables (PM2.5 and optional covariates)
3. Apply **MinMax scaling**
4. Convert data into sliding-window sequences
5. Define LSTM model using PyTorch
6. Train using MSE loss and Adam optimizer
7. Compute anomaly score:
   - `lstmscore` – prediction / reconstruction error
8. Apply thresholding:
   - percentile-based or statistical cutoff
9. Generate:
   - `lstmlabel` – binary anomaly flag
10. Save results to CSV

---

---

## 4. How the Unified Dataset Is Built

1. Align dates across AQI, fire, and weather datasets
2. Merge datasets to ensure **one row per day**
3. Handle missing values and outliers
4. Compute rolling PM2.5 features:
   - 7-day mean and standard deviation
5. Compute PM2.5 deviation from rolling mean
6. Assign season labels based on month
7. Attach AQI and AQI bucket information

The resulting dataset acts as the **canonical feature table** for all experiments.

---

## 6. Usage Guide

### 6.1 Prerequisites

- Python 3.x
- Core libraries:
  - pandas, numpy
  - scikit-learn
  - torch (PyTorch)
  - matplotlib / seaborn

---

### 6.2 Example Workflow

1. Prepare environment and install dependencies
2. Place datasets under `data/`
3. Perform exploratory data analysis
4. Run Isolation Forest pipeline
5. Run LSTM pipeline
6. Compare anomaly labels and scores
7. Visualize anomalies over PM2.5 and AQI time series

---

## 7. Possible Extensions

- Station-level (spatial) anomaly detection
- Multi-target LSTM with pollutants + weather
- Hybrid ensemble anomaly scoring
- Correlation with health or policy intervention data

---

## 8. Summary of Key Files

| File Name | Description |
|----------|--------------|
| `delhi_air_anomaly_ready.csv` | Cleaned, merged dataset with AQI, fire, weather, rolling PM2.5 stats |
| `delhi_air_anomaly_with_iforest_train_test.csv` | Isolation Forest anomaly scores and labels |
| `delhi_air_anomaly_with_lstm_minmax_pytorch.csv` | LSTM anomaly scores and labels |

---

This repository demonstrates a **comparative, real-world time-series anomaly detection framework** for environmental monitoring using both classical ML and deep learning approaches.








if i give u a dataset can u analyze it and tell what is the problem for project 📘 Project Description
Anomaly Detection in Air Quality Data for Early Identification of Pollution Events
🔴 Problem Statement

Air pollution has become a critical environmental and public health issue, especially in urban and industrial regions. Existing air quality monitoring systems primarily focus on reporting pollutant concentrations and Air Quality Index (AQI) values after they cross predefined threshold limits. Such rule-based systems are reactive in nature and often fail to provide early warnings for sudden and unexpected pollution events.

Pollution spikes caused by incidents such as industrial gas leaks, crop residue burning, accidental fires, sudden traffic congestion, or unfavorable meteorological conditions occur irregularly and do not always follow predictable patterns. These abnormal events are difficult to identify using traditional statistical methods or fixed thresholds, leading to delayed response from authorities and increased health risks for the population.

Therefore, there is a need for an intelligent system that can continuously analyze air quality data, learn normal pollution behavior over time, and automatically detect unusual deviations at an early stage without relying solely on predefined limits or manual intervention.

🟢 Proposed Solution

This project proposes an AI-based anomaly detection system for early identification of abnormal air pollution events using air quality time-series data. The system analyzes historical and real-time pollutant data such as PM2.5, PM10, NO₂, SO₂, CO, and O₃ to learn normal air quality patterns for a specific location and time period.

Machine learning and deep learning techniques are employed to model temporal variations in pollution levels. By learning what constitutes normal air behavior, the model can identify sudden spikes, irregular fluctuations, or unusual patterns as anomalies. These detected anomalies are flagged as potential pollution events, enabling early alerts.

To improve accuracy and reduce false alarms, meteorological parameters such as temperature, humidity, wind speed, and rainfall can be integrated into the model. Additionally, detected anomalies can be cross-validated with external event reference data such as satellite-based fire detection or reported industrial incidents to strengthen reliability and explanation.

The system outputs anomaly scores, timestamps, and visual indicators highlighting abnormal pollution periods. These results can be displayed through graphs or dashboards, supporting timely decision-making by environmental authorities, healthcare agencies, and smart city platforms.





How to move from “middle” to “goal”
To align fully with your original project description, next focus on:

Adding a sequence model (e.g., LSTM/GRU/Temporal CNN) trained on sliding windows of pollutants + meteorology to produce anomaly scores, then compare with Isolation Forest.

Designing a simple real‑time simulation: e.g., feed new days sequentially and trigger alerts when the anomaly score crosses a threshold.

Building a visual dashboard (Streamlit/Plotly): time‑series plots, anomaly markers, AQI buckets, and tooltips showing fire counts or weather context for each spike.

Doing case studies on known events (seasonal stubble burning, Diwali, extreme weather) to show your model detects them earlier than threshold-based AQI rules.