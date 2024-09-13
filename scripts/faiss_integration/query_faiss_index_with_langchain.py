from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
import numpy as np
import faiss

# load_dotenv()


# Function to query the FAISS index with a sample vector
def query_faiss_index_with_langchain(index_file, query_vector, top_k=5):
    print(f"Loading FAISS index from {index_file}...")

    # Load the FAISS index from disk
    index = faiss.read_index(index_file)
    print("FAISS index loaded successfully.")

    # Search for the nearest neighbors
    print(f"Querying FAISS index for top {top_k} results...")
    distances, indices = index.search(query_vector, top_k)

    print(f"Indices of nearest neighbors: {indices}")
    print(f"Distances to nearest neighbors: {distances}")

    return indices, distances


if __name__ == "__main__":
    # Example query vector (ensure it has the same dimension as your embeddings)
    query_vector = np.random.rand(1, 384).astype(
        np.float32
    )  # Replace with real query if needed

    # Path to the FAISS index files
    index_files = {
        "fitbit": "faiss_index_using_langchain/faiss_fitbit.index",
        "health": "faiss_index_using_langchain/faiss_health.index",
        "nutrition": "faiss_index_using_langchain/faiss_nutrition.index",
    }

    # Query FAISS index for each dataset
    for name, index_file in index_files.items():
        print(f"\nQuerying FAISS index for {name} data...")
        query_faiss_index_with_langchain(index_file, query_vector)
