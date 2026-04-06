import numpy as np
import pandas as pd

def generate_data():
    np.random.seed(42)
    data = np.random.normal(50, 5, 100)
    data[::10] += 20
    return pd.DataFrame({"sensor_value": data})

def detect_anomalies(df):
    threshold = df["sensor_value"].mean() + 2 * df["sensor_value"].std()
    df["anomaly"] = df["sensor_value"] > threshold
    return df