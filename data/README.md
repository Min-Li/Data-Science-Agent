# Data Directory

This directory contains:

## vector_db/
Pre-built vector database with embeddings for 647 Kaggle competitions.
- `embeddings.npy` - Numpy array of competition embeddings
- `metadata.json` - Competition details and solutions
- `index.faiss` - FAISS index for fast similarity search

## sample_datasets/
Example datasets for testing the agent:
- `iris.csv` - Classic classification dataset
- `titanic.csv` - Binary classification example
- `housing.csv` - Regression example
- `customer_churn.csv` - Business problem example

The vector database should be built using the existing Kaggle data processing scripts in the parent directory. 