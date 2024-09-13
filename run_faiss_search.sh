#!/bin/bash

# Start the FAISS Search API
echo "Starting the FAISS Search API..."
python3 api/faiss_search_api.py &

# Wait for the API to start
echo "Waiting for API to start..."
sleep 10  # Increase the sleep time to 10 seconds

# Generate a 384-dimensional query vector
query_vector=$(python3 -c "import numpy as np; print(np.random.rand(384).tolist())")

# Test Standard FAISS indices
echo "Testing FAISS search with Fitbit index (Standard FAISS)..."
curl -X POST http://127.0.0.1:5000/search -H "Content-Type: application/json" -d '{
  "index_type": "fitbit",
  "query_vector": '"$query_vector"',
  "top_k": 5
}'

# Testing Health index (Standard FAISS)
echo "Testing FAISS search with Health index (Standard FAISS)..."
curl -X POST http://127.0.0.1:5000/search -H "Content-Type: application/json" -d '{
  "index_type": "health",
  "query_vector": '"$query_vector"',
  "top_k": 5
}'

# Testing Nutrition index (Standard FAISS)
echo "Testing FAISS search with Nutrition index (Standard FAISS)..."
curl -X POST http://127.0.0.1:5000/search -H "Content-Type: application/json" -d '{
  "index_type": "nutrition",
  "query_vector": '"$query_vector"',
  "top_k": 5
}'

# Stop the FAISS Search API
echo "Stopping the FAISS Search API..."
kill $!
