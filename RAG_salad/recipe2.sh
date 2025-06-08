#!/bin/bash

# Create virtual environment
python3 -m venv reci2
source reci2/bin/activate

# Install required packages
pip install --upgrade pip
pip install streamlit llama-index llama-index-core boto3 requests
pip install llama-index-embeddings-bedrock llama-index-llms-bedrock llama-index-llms-bedrock-converse


echo "Setup completed successfully!"
