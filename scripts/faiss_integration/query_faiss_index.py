import faiss
import numpy as np
import os
import psutil


def monitor_memory():
    """Function to monitor memory usage."""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    print(f"Memory usage: {memory_info.rss / (1024 * 1024):.2f} MB")


def search_faiss_index(query_vector, index_file, top_k=5):
    """Search the FAISS index for the nearest neighbors to the query vector."""
    if not os.path.exists(index_file):
        raise FileNotFoundError(f"FAISS index file '{index_file}' not found.")

    print(f"Loading FAISS index from {index_file}...")
    index = faiss.read_index(index_file)
    monitor_memory()

    # Ensure the query vector is a NumPy array and reshape for FAISS
    query_vector = np.array(query_vector).reshape(1, -1)

    # Search the FAISS index
    print(f"Searching FAISS index for top {top_k} nearest neighbors...")
    distances, indices = index.search(query_vector, top_k)

    print(f"Nearest neighbor indices: {indices}")
    print(f"Nearest neighbor distances: {distances}")

    return indices, distances


if __name__ == "__main__":
    query_vector = np.random.rand(1, 384).astype(
        np.float32
    )  # Replace with real query if needed

    # Path to the FAISS index files
    index_files = {
        "fitbit": "faiss_index/faiss_fitbit.index",
        "health": "faiss_index/faiss_health.index",
        "nutrition": "faiss_index/faiss_nutrition.index",
    }

    # Query FAISS index for each dataset
    for name, index_file in index_files.items():
        print(f"\nQuerying FAISS index for {name} data...")
        search_faiss_index(query_vector, index_file, top_k=5)
