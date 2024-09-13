import pandas as pd
import json


# Function to load a CSV file into a pandas DataFrame
def load_csv_file(file_path):
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


# Function to convert a DataFrame to structured JSON format
def convert_to_json(df):
    try:
        return df.to_dict(orient="records")
    except Exception as e:
        print(f"Error converting DataFrame to JSON: {e}")
        return None


# Function to save JSON data to a file
def save_json(data, output_path):
    try:
        with open(output_path, "w") as json_file:
            json.dump(data, json_file, indent=4)
        print(f"Data successfully saved to {output_path}")
    except Exception as e:
        print(f"Error saving JSON to {output_path}: {e}")


# Function to process Fitbit data and convert to JSON
def process_data(file_paths, output_paths):
    for data_type, file_path in file_paths.items():
        print(f"Processing {data_type} data...")

        # Step 1: Load the CSV file
        df = load_csv_file(file_path)
        if df is not None:
            # Step 2: Convert to JSON format
            json_data = convert_to_json(df)
            if json_data:
                # Step 3: Save the JSON data to a file
                save_json(json_data, output_paths[data_type])


# Define the paths to your downloaded Fitbit CSV files (Update paths accordingly)
fitbit_file_paths = {
    "daily_activity": "dataset/mturkfitbit_export_4.12.16-5.12.16/Fitabase Data 4.12.16-5.12.16/dailyActivity_merged.csv",
    "heartrate": "dataset/mturkfitbit_export_4.12.16-5.12.16/Fitabase Data 4.12.16-5.12.16/heartrate_seconds_merged.csv",
    "sleep": "dataset/mturkfitbit_export_4.12.16-5.12.16/Fitabase Data 4.12.16-5.12.16/minuteSleep_merged.csv",
    "calories": "dataset/mturkfitbit_export_4.12.16-5.12.16/Fitabase Data 4.12.16-5.12.16/hourlyCalories_merged.csv",
}

# Define the paths where you want to save the JSON files (Update paths accordingly)
output_paths = {
    "daily_activity": "data/fitbit/daily_activity.json",
    "heartrate": "data/fitbit/heartrate.json",
    "sleep": "data/fitbit/sleep.json",
    "calories": "data/fitbit/calories.json",
}

# Define the paths to your USDA food data CSV files
food_file_paths = {
    "sr_legacy_food": "dataset/nutrition/sr_legacy_food.csv",
    "foundation_food": "dataset/nutrition/foundation_food.csv",
    "survey_food": "dataset/nutrition/survey_fndds_food.csv",
    "food_group_1": "dataset/nutrition/FINAL FOOD DATASET/FOOD-DATA-GROUP1.csv",
    "food_group_2": "dataset/nutrition/FINAL FOOD DATASET/FOOD-DATA-GROUP2.csv",
    "food_group_3": "dataset/nutrition/FINAL FOOD DATASET/FOOD-DATA-GROUP3.csv",
    "food_group_4": "dataset/nutrition/FINAL FOOD DATASET/FOOD-DATA-GROUP4.csv",
    "food_group_5": "dataset/nutrition/FINAL FOOD DATASET/FOOD-DATA-GROUP5.csv",
}

# Define the paths where you want to save the food data JSON files
food_output_paths = {
    "sr_legacy_food": "data/nutrition/sr_legacy_food.json",
    "foundation_food": "data/nutrition/foundation_food.json",
    "survey_food": "data/nutrition/survey_food.json",
    "food_group_1": "data/nutrition/food_group_1.json",
    "food_group_2": "data/nutrition/food_group_2.json",
    "food_group_3": "data/nutrition/food_group_3.json",
    "food_group_4": "data/nutrition/food_group_4.json",
    "food_group_5": "data/nutrition/food_group_5.json",
}

health_file_paths = {
    "health_data": "dataset/health/diabetes_012_health_indicators_BRFSS2015.csv",
}

# Define the paths where you want to save the JSON files (Update paths accordingly)
health_output_paths = {
    "health_data": "data/health/health_data.json",
}

process_data(fitbit_file_paths, output_paths)

process_data(food_file_paths, food_output_paths)

process_data(health_file_paths, health_output_paths)
