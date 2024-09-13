import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# Preprocess health data
def preprocess_health_data():
    health_data = pd.read_json("data/health/health_data.json")
    print("health_data.columns", health_data.columns)

    # Normalize the numeric columns (e.g., BMI, physical activity)
    scaler = MinMaxScaler()
    health_data[["BMI", "PhysActivity"]] = scaler.fit_transform(
        health_data[["BMI", "PhysActivity"]]
    )

    # Save preprocessed health data
    health_data.to_json("data/health/preprocessed_health_data.json", orient="records")


if __name__ == "__main__":
    preprocess_health_data()
