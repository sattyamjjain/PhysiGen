import pandas as pd


# Load the nutrition JSON files
def explore_nutrition_data():
    # Load Nutrition data
    food_group_1 = pd.read_json("data/nutrition/food_group_1.json")
    food_group_2 = pd.read_json("data/nutrition/food_group_2.json")
    food_group_3 = pd.read_json("data/nutrition/food_group_3.json")
    food_group_4 = pd.read_json("data/nutrition/food_group_4.json")
    food_group_5 = pd.read_json("data/nutrition/food_group_5.json")

    print("Food Group 1 Summary:")
    print(food_group_1.describe())

    print("Food Group 2 Summary:")
    print(food_group_2.describe())

    print("Food Group 3 Summary:")
    print(food_group_3.describe())

    print("Food Group 4 Summary:")
    print(food_group_4.describe())

    print("Food Group 5 Summary:")
    print(food_group_5.describe())


if __name__ == "__main__":
    explore_nutrition_data()
