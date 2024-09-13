import pandas as pd


# Load health data (diabetes, BMI indicators)
def explore_health_data():
    health_data = pd.read_json("data/health/health_data.json")

    # Check for missing values
    print("Health Data Missing Values:")
    print(health_data.isnull().sum())

    # View health data
    print("Health Data Preview:")
    print(health_data.head())


if __name__ == "__main__":
    explore_health_data()
