# AI-Based Predictive Maintenance System

<p align="center">
  <img src="https://img.shields.io/badge/Machine%20Learning-Pipeline-blue?style=flat-square" />
  <img src="https://img.shields.io/badge/Python-3.10+-yellow?style=flat-square" />
  <img src="https://img.shields.io/badge/Status-Active-success?style=flat-square" />
</p>

<p align="center">
  End-to-end pipeline for predictive maintenance using sensor data  
  Statistical anomaly detection for early fault identification
</p>

---

## Overview

This project implements a predictive maintenance pipeline that analyzes machine sensor data to identify abnormal operating conditions.

The system focuses on detecting deviations in sensor readings that may indicate potential failures. It is designed as a modular and extensible codebase that can be extended with more advanced machine learning techniques.

---

## System Architecture

The pipeline follows a structured workflow:

[Raw Data]
↓
[Data Processing]
↓
[Feature Selection]
↓
[Anomaly Detection]
↓
[Visualization / Output]

---

## Dataset

The pipeline operates on structured sensor data representing machine operating conditions. Typical features include:

- Air temperature [K]  
- Process temperature [K]  
- Rotational speed [rpm]  
- Torque [Nm]  
- Tool wear [min]  

These features are combined to derive a representative signal used for anomaly detection.

---

## Methodology

The current implementation uses a statistical approach to detect anomalies.

Execution flow:

1. Load dataset into memory  
2. Select relevant sensor features  
3. Aggregate multiple sensor readings into a single signal  
4. Compute mean and standard deviation of the signal  
5. Define upper and lower bounds using a threshold factor  
6. Flag observations outside these bounds as anomalies  

This approach provides a simple and interpretable baseline for identifying abnormal behavior.

---

## Implementation

The anomaly detection module supports multivariate input by aggregating sensor features:

```python
def detect_anomalies(df, sensor_columns=None, threshold=3):
    if sensor_columns is None:
        sensor_columns = [
            'Air temperature [K]',
            'Process temperature [K]',
            'Rotational speed [rpm]',
            'Torque [Nm]',
            'Tool wear [min]'
        ]

    df = df.copy()

    df['sensor_value'] = df[sensor_columns].mean(axis=1)

    mean = df['sensor_value'].mean()
    std = df['sensor_value'].std()

    lower_bound = mean - threshold * std
    upper_bound = mean + threshold * std

    anomalies = df[
        (df['sensor_value'] < lower_bound) |
        (df['sensor_value'] > upper_bound)
    ]

    return anomalies, df

This design avoids reliance on a single column and allows the system to adapt to different datasets.

## Project Structure

predictive-maintenance-ml-pipeline/
│
├── data/                      # Dataset (raw or processed)
├── notebooks/                 # Exploratory analysis (optional)
├── src/
│   ├── main.py                # Entry point for pipeline execution
│   ├── anomaly_detection.py   # Core detection logic
│
├── requirements.txt
└── README.md


---

---

## Execution

The pipeline operates as follows:

1. Load the dataset into memory  
2. Aggregate selected sensor features into a unified signal  
3. Compute statistical boundaries based on mean and standard deviation  
4. Identify observations outside the defined range as anomalies  
5. Visualize the signal along with detected anomalies  

---

## Output

The system produces a time-series visualization of the sensor signal:

- Continuous signal represents normal operating conditions  
- Points outside statistical bounds are marked as anomalies  

This allows quick inspection of abnormal behavior in the data.

---

## Running the Project

Install dependencies:

```bash
pip install -r requirements.txt

---

## How to Run

1. Install dependencies:

pip install numpy pandas matplotlib

2. Run the script:

python src/main.py

---

## Future Improvements
- Replace statistical method with machine learning models (Isolation Forest, Autoencoders)
- Introduce time-series modeling for sequential data
- Add real-time data ingestion and monitoring  
- Expose functionality via an API (FastAPI)
- Integrate experiment tracking (MLflow) 
- Containerize the application using Docker

---

## Author
Md Abrar FahimMd Abrar Fahim
B.Sc. Information Engineering
HAW Hamburg
Working Student – Data & AI