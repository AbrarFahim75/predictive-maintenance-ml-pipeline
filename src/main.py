import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

from anomaly_detection import detect_anomalies, generate_data, load_sensor_csv

# Default path if you place the AI4I 2020 CSV under project root / data
DEFAULT_CSV = Path(__file__).resolve().parent.parent / "data" / "ai4i2020.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Anomaly detection on sensor data (CSV or synthetic)."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Path to CSV file (defaults to data/ai4i2020.csv if present).",
    )
    parser.add_argument(
        "--column",
        default="Torque [Nm]",
        help="Numeric column to treat as the sensor (AI4I default: Torque [Nm]).",
    )
    parser.add_argument(
        "--time-column",
        default=None,
        help="Optional column for x-axis (e.g. timestamp or sample id).",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use built-in synthetic data instead of a file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.synthetic:
        df = generate_data()
    elif args.csv is not None:
        df = load_sensor_csv(
            str(args.csv),
            args.column,
            time_column=args.time_column,
        )
    elif DEFAULT_CSV.is_file():
        df = load_sensor_csv(
            str(DEFAULT_CSV),
            args.column,
            time_column=args.time_column,
        )
    else:
        raise SystemExit(
            f"No dataset found. Place ai4i2020.csv at {DEFAULT_CSV}, "
            "pass --csv PATH, or use --synthetic."
        )

    df = detect_anomalies(df)

    print(df.head(20))
    print()
    print(f"Anomalies detected: {df['anomaly'].sum()} / {len(df)}")

    x = df["time"] if "time" in df.columns else df.index
    plt.figure(figsize=(10, 4))
    plt.plot(x, df["sensor_value"], color="steelblue", linewidth=0.8, label="Sensor")
    anom = df["anomaly"]
    plt.scatter(
        x[anom],
        df.loc[anom, "sensor_value"],
        color="crimson",
        s=12,
        zorder=5,
        label="Anomaly",
    )
    plt.xlabel("Time / index")
    plt.ylabel(args.column if not args.synthetic else "sensor_value")
    plt.title("Anomaly detection (mean + 2σ threshold)")
    plt.legend()
    plt.tight_layout()
    if matplotlib.get_backend().lower() == "agg":
        out = Path(__file__).resolve().parent.parent / "output" / "anomaly_plot.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=120)
        print(f"Plot saved to {out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
