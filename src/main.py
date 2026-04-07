from data_loader import load_csv_data
from anomaly_detection import detect_anomalies

def main():
    file_path = "data/ai4i2020.csv"

    # Load real dataset
    df = load_csv_data(file_path)

    print("Dataset loaded successfully")
    print(df.head())

    # Select one column for now (keep simple)
    data = df["Air temperature [K]"]

    anomalies = detect_anomalies(data)

    print(f"Total anomalies detected: {len(anomalies)}")

if __name__ == "__main__":
    main()