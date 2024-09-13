from flask import Flask, request, jsonify
from langchain_community.vectorstores import FAISS as LangchainFAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
import faiss
import numpy as np

app = Flask(__name__)

# Index files for Standard FAISS
index_files = {
    "fitbit": "faiss_index/faiss_fitbit.index",
    "health": "faiss_index/faiss_health.index",
    "nutrition": "faiss_index/faiss_nutrition.index",
}

# Index files for Langchain FAISS
langchain_index_files = {
    "fitbit": "faiss_index_using_langchain/faiss_fitbit.faiss",
    "health": "faiss_index_using_langchain/faiss_health.faiss",
    "nutrition": "faiss_index_using_langchain/faiss_nutrition.faiss",
}

# Loaded FAISS indices
faiss_indices = {}
langchain_indices = {}


# Load Standard FAISS indices
def load_standard_faiss_indices():
    print("Loading FAISS indices...")
    for name, file in index_files.items():
        print(f"Loading FAISS index for {name} from {file}...")
        faiss_indices[name] = faiss.read_index(file)
        print(f"Loaded FAISS index for {name}")


# Load Langchain FAISS indices
def load_langchain_faiss_indices():
    print("Loading Langchain FAISS indices...")

    for name, file in langchain_index_files.items():
        print(f"Loading Langchain FAISS index for {name} from {file}...")
        # Directly load the FAISS index file
        faiss_index = faiss.read_index(file)  # Load the .faiss file
        langchain_indices[name] = LangchainFAISS(
            embedding_function=None,  # No embedding function needed for loading
            index=faiss_index,
            docstore=InMemoryDocstore(
                {}
            ),  # You can provide the real docstore here if needed
            index_to_docstore_id={},  # You can provide the real mapping if needed
        )
        print(f"Langchain FAISS index for {name} loaded successfully.")


@app.route("/search", methods=["POST"])
def search_faiss():
    try:
        data = request.json
        index_type = data.get("index_type", "fitbit")
        query_vector = np.array(data.get("query_vector", []), dtype=np.float32).reshape(
            1, -1
        )
        top_k = data.get("top_k", 5)

        if index_type in faiss_indices:
            index = faiss_indices[index_type]
        elif index_type in langchain_indices:
            index = langchain_indices[index_type].index
        else:
            return jsonify({"error": f"Unknown index type: {index_type}"}), 400

        if query_vector.shape[1] != index.d:
            return (
                jsonify(
                    {
                        "error": f"Query vector dimension {query_vector.shape[1]} does not match index dimension {index.d}"
                    }
                ),
                400,
            )

        distances, indices = index.search(query_vector, top_k)
        return jsonify({"distances": distances.tolist(), "indices": indices.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    load_standard_faiss_indices()
    load_langchain_faiss_indices()
    app.run(host="0.0.0.0", port=5000)
