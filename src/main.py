import numpy as np
import pandas as pd

# simulate sensor data
np.random.seed(42)
data = np.random.normal(loc=50, scale=5, size=100)

# add anomalies
data[::10] = data[::10] + 20

df = pd.DataFrame({"sensor_value": data})

# simple anomaly detection
threshold = df["sensor_value"].mean() + 2 * df["sensor_value"].std()
df["anomaly"] = df["sensor_value"] > threshold

print(df.head(20))