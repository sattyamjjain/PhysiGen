import faiss
import numpy as np
import time


# Function to test FAISS on CPU
def test_faiss_index():
    print("Testing FAISS on CPU...")

    # Generate random embeddings for testing
    num_embeddings = 10000
    embedding_dim = 128
    print(
        f"Generating {num_embeddings} random embeddings with dimension {embedding_dim}..."
    )
    embeddings = np.random.random((num_embeddings, embedding_dim)).astype("float32")

    # Create a FAISS index on CPU
    print("Using CPU for FAISS index...")
    start_time = time.time()
    index = faiss.IndexFlatL2(embedding_dim)  # L2 distance index

    # Add embeddings to the index
    print(f"Adding {num_embeddings} embeddings to the FAISS index...")
    index.add(embeddings)
    print(
        f"Embeddings added successfully. Time taken: {time.time() - start_time} seconds"
    )

    # Perform a search for the nearest neighbors of a random query vector
    print("Performing search for the nearest neighbors of a random query vector...")
    query_vector = np.random.random((1, embedding_dim)).astype("float32")
    k = 5  # Number of nearest neighbors to search for

    start_time = time.time()
    distances, indices = index.search(query_vector, k)
    print(f"Nearest neighbor indices: {indices}")
    print(f"Distances to nearest neighbors: {distances}")
    print(f"Search completed. Time taken: {time.time() - start_time} seconds")


if __name__ == "__main__":
    test_faiss_index()
