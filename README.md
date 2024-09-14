# PhysiGen Project

## Overview
PhysiGen is a comprehensive project designed to integrate FAISS indexing and Langchain for efficient vector search and embeddings management. This project processes and manages data from Fitbit, Health, and Nutrition datasets to generate embeddings, build FAISS indexes, and expose REST APIs for querying embeddings.

### Key Features:

- **Multi-source Embeddings Generation:** From Fitbit, Health, and Nutrition datasets.
- **Flexible FAISS Indexing:** Supports both standard FAISS and Langchain-based FAISS integrations.
- **Scalable REST APIs:** For querying generated embeddings from FAISS indexes.
- **Modular Code Structure:** With dedicated scripts for data preprocessing, embeddings generation, and FAISS index building.


## Setup Instructions

### Prerequisites
Ensure you have the following prerequisites installed before running the project:

- Python (3.9 or later)
- FAISS (Facebook AI Similarity Search library)
- Langchain and Langchain-Community (For FAISS and embeddings integration)

For GPU support:

- CUDA (if using FAISS with GPU)
- FAISS with GPU support


## Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/sattyamjjain/PhysiGen.git
   cd GenCore

2. Create a virtual environment and activate it:

   ```bash
    conda create -n PhysiGen python=3.11
    conda activate PhysiGen

3. Install the required dependencies:

   ```bash
    pip install -r requirements.txt

For GPU acceleration with FAISS, ensure your system has the correct CUDA toolkit installed.

## Data Preprocessing

Before generating embeddings, ensure all data is preprocessed. Run the following scripts for the respective datasets:

    # Fitbit data preprocessing
    python3 scripts/data_preprocessing/fitbit_data_preprocessing.py
    
    # Health data preprocessing
    python3 scripts/data_preprocessing/health_data_preprocessing.py
    
    # Nutrition data preprocessing
    python3 scripts/data_preprocessing/nutrition_data_preprocessing.py


## Embeddings Generation

Once the data is preprocessed, generate the embeddings:

    python3 scripts/embedding/embedding_generation.py

This will generate and store the embeddings in the embeddings/ folder.

## FAISS Index Creation

### Standard FAISS Index Creation

To create FAISS indexes:

    python3 scripts/faiss_integration/build_faiss_index.py

### Langchain-based FAISS Index Creation

To build FAISS indexes using Langchain:

    python3 scripts/faiss_integration/build_faiss_index_with_langchain.py

Both standard FAISS and Langchain-based FAISS indexes will be saved in their respective folders (faiss_index/ and faiss_index_using_langchain/).

## Querying the FAISS Index

### Standard FAISS Query

To query a FAISS index using standard FAISS:

    python3 scripts/faiss_integration/query_faiss_index.py

### Langchain-based FAISS Query

To query FAISS indexes built using Langchain:

    python3 scripts/faiss_integration/query_faiss_index_with_langchain.py

## Running the FAISS API

### 1. Starting the API

Start the FAISS Search API which handles both standard and Langchain FAISS:

    bash run_faiss_search.sh

This script starts the API server and runs a series of tests against the indexes. It will be available at http://127.0.0.1:5000.

### 2. Example Query

Once the API is running, you can send POST requests to query the FAISS index.

Example of querying the API:


    curl -X POST http://127.0.0.1:5000/search \
    -H "Content-Type: application/json" \
    -d '{
      "index_type": "fitbit",
      "query_vector": [0.5, 0.2, ...],  # Replace with an actual query vector
      "top_k": 5
    }'

The API will return the top 5 results from the requested FAISS index.

## Testing the Project

Several test scripts are provided:

    python3 samples/test_setup.py  # Test Base Setup
    python3 samples/test_faiss_cpu.py  # Test FAISS on CPU
    python3 samples/test_faiss_gpu.py  # Test FAISS on GPU

## Documentation

Find the detailed documentation in the docs/ folder.

## Future Enhancements

- **Integration with Pinecone or Weaviate:** For scalable and cloud-based vector search.
- **Streamlined Dataset Addition:** Add more datasets beyond Fitbit, Health, and Nutrition.
- **Advanced Query Options:** Support additional query types such as cosine similarity or approximate nearest neighbor (ANN) searches.
- **Real-time Search:** Implement WebSocket-based real-time querying.

## Contributors

- Sattyam Jain
- Contributions are welcome! Feel free to fork the repository and create pull requests for new features, bug fixes, or documentation improvements.

## Contact Information

For any questions or inquiries, reach out to Sattyam Jain at:

- Email: sattyamjain96@gmail.com
- LinkedIn: https://www.linkedin.com/in/sattyamjain/