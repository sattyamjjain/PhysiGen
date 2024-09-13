import faiss
import numpy as np
import time
import torch


# Function to test FAISS index on GPU
def test_faiss_index_on_gpu():
    if not torch.cuda.is_available():
        print("GPU is not available. Exiting...")
        return

    num_embeddings = 10000
    embedding_dim = 128
    k = 5  # Number of nearest neighbors

    print(
        f"Generating {num_embeddings} random embeddings with dimension {embedding_dim}..."
    )
    embeddings = np.random.random((num_embeddings, embedding_dim)).astype("float32")

    print("Using GPU for FAISS index...")
    res = faiss.StandardGpuResources()  # Initialize GPU resources

    # Create a GPU-based FAISS index (using L2 distance)
    index_flat = faiss.IndexFlatL2(embedding_dim)
    gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)

    # Add embeddings to the GPU index
    start_time = time.time()
    gpu_index_flat.add(embeddings)
    print(
        f"Embeddings added successfully on GPU. Time taken: {time.time() - start_time} seconds"
    )

    # Search for nearest neighbors on GPU
    print("Performing search for nearest neighbors on GPU...")
    query_vector = np.random.random((1, embedding_dim)).astype("float32")
    start_time = time.time()
    distances, indices = gpu_index_flat.search(query_vector, k)
    print(f"Nearest neighbor indices: {indices}")
    print(f"Distances to nearest neighbors: {distances}")
    print(f"Search completed on GPU. Time taken: {time.time() - start_time} seconds")


if __name__ == "__main__":
    test_faiss_index_on_gpu()
