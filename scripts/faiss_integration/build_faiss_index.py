import faiss
import numpy as np
import os
import time
import psutil


def monitor_memory():
    """Function to monitor memory usage."""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    print(f"Memory usage: {memory_info.rss / (1024 * 1024):.2f} MB")


def build_faiss_index(embedding_file, index_file):
    """Builds FAISS index for the given embedding file and saves the index."""
    print(f"Loading embeddings from {embedding_file}...")
    start_time = time.time()

    # Load embeddings
    embeddings = np.load(embedding_file)
    print(f"Embeddings loaded with shape: {embeddings.shape}")
    monitor_memory()

    # Get dimension of embeddings
    dim = embeddings.shape[1]

    # Initialize FAISS index (using L2 distance)
    print("Initializing FAISS index...")
    index = faiss.IndexFlatL2(dim)

    # Add embeddings to FAISS index
    print("Adding embeddings to FAISS index...")
    index.add(embeddings)
    print(f"Embeddings added. Time taken: {time.time() - start_time:.2f} seconds")
    monitor_memory()

    # Save FAISS index
    os.makedirs(os.path.dirname(index_file), exist_ok=True)
    print(f"Saving FAISS index to {index_file}...")
    faiss.write_index(index, index_file)
    print("FAISS index saved successfully.")


if __name__ == "__main__":
    os.makedirs("faiss_index", exist_ok=True)

    # Build FAISS index for all embeddings
    embedding_files = {
        "fitbit": "embeddings/fitbit_embeddings.npy",
        "health": "embeddings/health_embeddings.npy",
        "nutrition": "embeddings/nutrition_embeddings.npy",
    }

    index_files = {
        "fitbit": "faiss_index/faiss_fitbit.index",
        "health": "faiss_index/faiss_health.index",
        "nutrition": "faiss_index/faiss_nutrition.index",
    }

    for name, emb_file in embedding_files.items():
        print(f"\nBuilding FAISS index for {name} data...")
        build_faiss_index(emb_file, index_files[name])
