#!/bin/bash

# Game24 Evaluation Script
# This script evaluates the open source Game24 models on the test dataset

set -e

# Check if models are downloaded
if [ ! -d "models/ct2_cache/llama2-7b-game24-policy-ct2" ] || [ ! -d "models/value/llama2-7b-game24-value" ]; then
    echo "Error: Models not found. Please run setup_game24_evaluation.sh first."
    exit 1
fi

# Activate conda environment
source activate tsllm

# Set model paths
CT2_DIR="$(pwd)/models/ct2_cache/llama2-7b-game24-policy-ct2"
CRITIC_PATH="$(pwd)/models/value/llama2-7b-game24-value"
SAVE_DIR="$(pwd)/results"

echo "Starting Game24 evaluation..."
echo "CT2 Model: $CT2_DIR"
echo "Critic Model: $CRITIC_PATH"
echo "Results will be saved to: $SAVE_DIR"

# Set evaluation flags
export TEST_NO_TERMINAL=1

# Create results directory
mkdir -p $SAVE_DIR

# Run evaluation with different methods
echo "Running MCTS evaluation..."
torchrun --nproc_per_node=1 --master-port 29522 tsllm/offline_rl/test_sft_and_v.py \
    --critic_model_path $CRITIC_PATH \
    --tokenizer_path $CRITIC_PATH \
    --ct2_dir $CT2_DIR \
    --save_dir $SAVE_DIR/mcts_results \
    --env_name game24 \
    --test True

echo "Running Chain-of-Thought Self-Consistency evaluation..."
export TEST_NO_TERMINAL=0
export TEST_COT_SC=1

torchrun --nproc_per_node=1 --master-port 29523 tsllm/offline_rl/test_sft_and_v.py \
    --critic_model_path $CRITIC_PATH \
    --tokenizer_path $CRITIC_PATH \
    --ct2_dir $CT2_DIR \
    --save_dir $SAVE_DIR/cot_sc_results \
    --env_name game24 \
    --test True

echo "Running Chain-of-Thought Greedy evaluation..."
export TEST_COT_SC=0
export TEST_COT_GREEDY=1

torchrun --nproc_per_node=1 --master-port 29524 tsllm/offline_rl/test_sft_and_v.py \
    --critic_model_path $CRITIC_PATH \
    --tokenizer_path $CRITIC_PATH \
    --ct2_dir $CT2_DIR \
    --save_dir $SAVE_DIR/cot_greedy_results \
    --env_name game24 \
    --test True

echo "Evaluation completed!"
echo "Results saved to: $SAVE_DIR"
echo ""
echo "To format results for Kaggle submission, run: python format_kaggle_submission.py"