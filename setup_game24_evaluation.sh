#!/bin/bash

# Game24 Model Evaluation Setup Script
# This script sets up the environment and downloads models for Game24 evaluation

set -e

echo "Setting up Game24 evaluation environment..."

# Create directories for models and results
mkdir -p models/policy
mkdir -p models/value
mkdir -p models/ct2_cache
mkdir -p results
mkdir -p kaggle_submission

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Error: conda is required but not installed. Please install conda first."
    exit 1
fi

# Check if environment exists
if conda env list | grep -q tsllm; then
    echo "Environment 'tsllm' already exists. Activating..."
    source activate tsllm
else
    echo "Creating conda environment 'tsllm'..."
    conda create -n tsllm python=3.10 -y
    source activate tsllm
fi

echo "Installing requirements..."
pip install -r requirement.txt
pip install -e .

# Additional packages for evaluation and Kaggle submission
pip install huggingface_hub kaggle pandas

echo "Downloading pre-trained Game24 models..."

# Download policy model
if [ ! -d "models/policy/llama2-7b-game24-policy-hf" ]; then
    echo "Downloading Game24 policy model..."
    cd models/policy
    git clone https://huggingface.co/OhCherryFire/llama2-7b-game24-policy-hf
    cd ../..
else
    echo "Policy model already exists, skipping download..."
fi

# Download value model
if [ ! -d "models/value/llama2-7b-game24-value" ]; then
    echo "Downloading Game24 value model..."
    cd models/value
    git clone https://huggingface.co/OhCherryFire/llama2-7b-game24-value
    cd ../..
else
    echo "Value model already exists, skipping download..."
fi

echo "Converting policy model to CTranslate2 format..."
if [ ! -d "models/ct2_cache/llama2-7b-game24-policy-ct2" ]; then
    ct2-transformers-converter \
        --model models/policy/llama2-7b-game24-policy-hf \
        --quantization bfloat16 \
        --output_dir models/ct2_cache/llama2-7b-game24-policy-ct2
    echo "Policy model converted to CTranslate2 format."
else
    echo "CTranslate2 policy model already exists, skipping conversion..."
fi

echo "Setup completed successfully!"
echo ""
echo "Models downloaded to:"
echo "  - Policy model: models/policy/llama2-7b-game24-policy-hf"
echo "  - Value model: models/value/llama2-7b-game24-value"
echo "  - CT2 cache: models/ct2_cache/llama2-7b-game24-policy-ct2"
echo ""
echo "To run evaluation, use: ./evaluate_game24.sh"
echo "To activate the environment: conda activate tsllm"