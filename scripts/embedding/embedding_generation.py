import os

import pandas as pd
from sentence_transformers import SentenceTransformer
import torch
import numpy as np

# Initialize the sentence transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
model = model.to(device)

os.makedirs("embeddings", exist_ok=True)


# Function to generate embeddings for Fitbit data in batches
def generate_fitbit_embeddings(batch_size=10000):
    print("Step 1: Loading Fitbit data...")
    daily_activity = pd.read_json("data/fitbit/preprocessed_daily_activity.json")
    heartrate = pd.read_json("data/fitbit/preprocessed_heartrate.json")
    sleep = pd.read_json("data/fitbit/preprocessed_sleep.json")
    calories = pd.read_json("data/fitbit/preprocessed_calories.json")

    print(
        f"Fitbit data loaded. Daily activity shape: {daily_activity.shape}, Heartrate shape: {heartrate.shape}, Sleep shape: {sleep.shape}, Calories shape: {calories.shape}"
    )

    # Combine all relevant columns into a single text representation
    print("Step 2: Combining data...")
    fitbit_data = pd.concat([daily_activity, heartrate, sleep, calories], axis=1)

    # Convert all numeric values to strings to avoid errors
    fitbit_data = fitbit_data.astype(str)
    print(f"Fitbit data combined. Final shape: {fitbit_data.shape}")

    # Process data in batches to reduce memory consumption
    embeddings = []
    total_rows = fitbit_data.shape[0]
    print(
        f"Step 3: Generating embeddings for Fitbit data in batches of {batch_size}..."
    )

    for i in range(0, total_rows, batch_size):
        print(
            f"Processing batch {i // batch_size + 1} / {total_rows // batch_size + 1}"
        )
        batch = fitbit_data.iloc[i : i + batch_size].values.tolist()
        batch_embeddings = model.encode(batch, convert_to_tensor=False, device=device)
        embeddings.extend(batch_embeddings)

    # Save the embeddings
    np.save("embeddings/fitbit_embeddings.npy", embeddings)
    print("Fitbit embeddings saved.")


# Function to generate embeddings for health data in batches
def generate_health_embeddings(batch_size=10000):
    print("Step 1: Loading health data...")
    health_data = pd.read_json("data/health/preprocessed_health_data.json")

    # Convert all numeric values to strings to avoid errors
    health_data = health_data.astype(str)

    print(f"Health data loaded. Shape: {health_data.shape}")

    # Process data in batches to reduce memory consumption
    embeddings = []
    total_rows = health_data.shape[0]
    print(
        f"Step 2: Generating embeddings for health data in batches of {batch_size}..."
    )

    for i in range(0, total_rows, batch_size):
        print(
            f"Processing batch {i // batch_size + 1} / {total_rows // batch_size + 1}"
        )
        batch = health_data.iloc[i : i + batch_size].values.tolist()
        batch_embeddings = model.encode(batch, convert_to_tensor=False, device=device)
        embeddings.extend(batch_embeddings)

    # Save the embeddings
    np.save("embeddings/health_embeddings.npy", embeddings)
    print("Health embeddings saved.")


# Function to generate embeddings for nutrition data in batches
def generate_nutrition_embeddings(batch_size=10000):
    print("Step 1: Loading nutrition data...")
    nutrition_data = []
    for i in range(1, 6):
        nutrition_data.append(
            pd.read_json(f"data/nutrition/preprocessed_food_group_{i}.json").astype(str)
        )

    nutrition_data = pd.concat(nutrition_data, axis=0)
    print(f"Nutrition data loaded. Shape: {nutrition_data.shape}")

    # Process data in batches to reduce memory consumption
    embeddings = []
    total_rows = nutrition_data.shape[0]
    print(
        f"Step 2: Generating embeddings for nutrition data in batches of {batch_size}..."
    )

    for i in range(0, total_rows, batch_size):
        print(
            f"Processing batch {i // batch_size + 1} / {total_rows // batch_size + 1}"
        )
        batch = nutrition_data.iloc[i : i + batch_size].values.tolist()
        batch_embeddings = model.encode(batch, convert_to_tensor=False, device=device)
        embeddings.extend(batch_embeddings)

    # Save the embeddings
    np.save("embeddings/nutrition_embeddings.npy", embeddings)
    print("Nutrition embeddings saved.")


if __name__ == "__main__":
    print("Starting Fitbit embedding generation...")
    generate_fitbit_embeddings(batch_size=10000)

    print("Starting Health embedding generation...")
    generate_health_embeddings(batch_size=10000)

    print("Starting Nutrition embedding generation...")
    generate_nutrition_embeddings(batch_size=10000)
