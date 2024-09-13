import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# Load and preprocess Nutrition data
def preprocess_nutrition_data():
    # Load Nutrition data files individually
    food_group_1 = pd.read_json("data/nutrition/food_group_1.json")
    food_group_2 = pd.read_json("data/nutrition/food_group_2.json")
    food_group_3 = pd.read_json("data/nutrition/food_group_3.json")
    food_group_4 = pd.read_json("data/nutrition/food_group_4.json")
    food_group_5 = pd.read_json("data/nutrition/food_group_5.json")

    # Initialize MinMaxScaler for normalization
    scaler = MinMaxScaler()

    # Preprocess and normalize relevant columns for food group data
    for food_group, name in zip(
        [food_group_1, food_group_2, food_group_3, food_group_4, food_group_5],
        [
            "food_group_1",
            "food_group_2",
            "food_group_3",
            "food_group_4",
            "food_group_5",
        ],
    ):
        if "Caloric Value" in food_group.columns and "Fat" in food_group.columns:
            food_group[["Caloric Value", "Fat"]] = scaler.fit_transform(
                food_group[["Caloric Value", "Fat"]]
            )
        food_group.to_json(f"data/nutrition/preprocessed_{name}.json", orient="records")
        print(f"Preprocessed {name} saved.")


if __name__ == "__main__":
    preprocess_nutrition_data()
