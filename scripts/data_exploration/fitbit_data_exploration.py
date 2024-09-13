import pandas as pd


# Load the Fitbit JSON files
def explore_fitbit_data():
    # Load Fitbit data
    daily_activity = pd.read_json("data/fitbit/daily_activity.json")
    heartrate = pd.read_json("data/fitbit/heartrate.json")
    sleep = pd.read_json("data/fitbit/sleep.json")
    calories = pd.read_json("data/fitbit/calories.json")

    # Display summary statistics for each dataset
    print("Daily Activity Summary:")
    print(daily_activity.describe())

    print("Heartrate Summary:")
    print(heartrate.describe())

    print("Sleep Summary:")
    print(sleep.describe())

    print("Calories Summary:")
    print(calories.describe())


if __name__ == "__main__":
    explore_fitbit_data()
