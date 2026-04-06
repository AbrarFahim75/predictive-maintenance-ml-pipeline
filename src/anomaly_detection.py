import numpy as np
import pandas as pd
from typing import Optional


def generate_data(n: int = 100, seed: int = 42) -> pd.DataFrame:
    """Synthetic demo data (optional fallback for testing)."""
    np.random.seed(seed)
    data = np.random.normal(50, 5, n)
    data[::10] += 20
    return pd.DataFrame({"sensor_value": data})


def load_sensor_csv(
    path: str,
    value_column: str,
    *,
    time_column: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load a CSV and expose one numeric column as ``sensor_value`` for detection.

    ``time`` is set from ``time_column`` if given; otherwise a simple row index 0..n-1.
    """
    raw = pd.read_csv(path)
    if value_column not in raw.columns:
        raise ValueError(
            f"Column {value_column!r} not found. Available: {list(raw.columns)}"
        )
    values = pd.to_numeric(raw[value_column], errors="coerce")
    out = pd.DataFrame({"sensor_value": values})
    if time_column is not None and time_column in raw.columns:
        out["time"] = raw[time_column]
    else:
        out["time"] = np.arange(len(out), dtype=float)
    return out.dropna(subset=["sensor_value"]).reset_index(drop=True)


def detect_anomalies(df: pd.DataFrame, value_column: str = "sensor_value") -> pd.DataFrame:
    """Mark rows above mean + 2 * std as anomalies (global threshold)."""
    series = df[value_column]
    threshold = series.mean() + 2 * series.std()
    result = df.copy()
    result["anomaly"] = series > threshold
    return result
