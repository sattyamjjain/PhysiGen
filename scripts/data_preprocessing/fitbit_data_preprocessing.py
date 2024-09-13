import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# Load and preprocess Fitbit data
def preprocess_fitbit_data():
    # Load Fitbit data
    daily_activity = pd.read_json("data/fitbit/daily_activity.json")
    heartrate = pd.read_json("data/fitbit/heartrate.json")
    sleep = pd.read_json("data/fitbit/sleep.json")
    calories = pd.read_json("data/fitbit/calories.json")
    print("daily_activity.columns", daily_activity.columns)
    print("heartrate.columns", heartrate.columns)
    print("sleep.columns", sleep.columns)
    print("calories.columns", calories.columns)

    # Normalize the numeric columns (e.g., steps, calories)
    scaler = MinMaxScaler()

    daily_activity[["TotalSteps", "Calories"]] = scaler.fit_transform(
        daily_activity[["TotalSteps", "Calories"]]
    )
    heartrate["Value"] = scaler.fit_transform(
        heartrate[["Value"]]
    )  # Correct column name 'Value'
    sleep["value"] = scaler.fit_transform(sleep[["value"]])
    calories["Calories"] = scaler.fit_transform(calories[["Calories"]])

    # Save the preprocessed data
    daily_activity.to_json(
        "data/fitbit/preprocessed_daily_activity.json", orient="records"
    )
    heartrate.to_json("data/fitbit/preprocessed_heartrate.json", orient="records")
    sleep.to_json("data/fitbit/preprocessed_sleep.json", orient="records")
    calories.to_json("data/fitbit/preprocessed_calories.json", orient="records")


if __name__ == "__main__":
    preprocess_fitbit_data()
