from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
import numpy as np
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.schema import Document
import os

load_dotenv()


# Function to build FAISS index directly with embeddings
def build_faiss_index_with_langchain(embedding_file, index_file):
    try:
        print(f"Loading embeddings from {embedding_file}...")

        # Load embeddings from the saved file
        embeddings = np.load(embedding_file)
        print(f"Embeddings loaded with shape: {embeddings.shape}")

        # Initialize FAISS index with the dimension of the embeddings
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)  # L2 distance for similarity search

        # Add embeddings to the FAISS index
        print(f"Adding embeddings to the FAISS index...")
        index.add(embeddings)

        # Create dummy documents and IDs for the embedding vectors
        docs = [Document(page_content=f"Document {i}") for i in range(len(embeddings))]
        docstore = InMemoryDocstore({str(i): docs[i] for i in range(len(embeddings))})
        index_to_docstore_id = {i: str(i) for i in range(len(embeddings))}

        # Use Langchain FAISS wrapper to build the vector store
        vector_store = FAISS(
            embedding_function=None,  # No embedding function needed
            index=index,
            docstore=docstore,
            index_to_docstore_id=index_to_docstore_id,
        )

        # Save the FAISS index using FAISS native serialization
        print(f"Saving the FAISS index to {index_file} using FAISS write_index...")
        os.makedirs(os.path.dirname(index_file), exist_ok=True)
        faiss.write_index(index, index_file)  # Save the FAISS index natively
        print(f"FAISS index saved to {index_file}")

    except Exception as e:
        print(f"Error occurred: {e}")


if __name__ == "__main__":
    # Build FAISS index for all embeddings
    embedding_files = {
        "fitbit": "embeddings/fitbit_embeddings.npy",
        "health": "embeddings/health_embeddings.npy",
        "nutrition": "embeddings/nutrition_embeddings.npy",
    }

    index_files = {
        "fitbit": "faiss_index_using_langchain/faiss_fitbit.faiss",  # Changed to .faiss
        "health": "faiss_index_using_langchain/faiss_health.faiss",  # Changed to .faiss
        "nutrition": "faiss_index_using_langchain/faiss_nutrition.faiss",  # Changed to .faiss
    }

    for name, emb_file in embedding_files.items():
        print(f"\nBuilding FAISS index for {name} data...")
        build_faiss_index_with_langchain(emb_file, index_files[name])
