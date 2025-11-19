# Game24 Open Source Model Evaluation for Kaggle

This guide will help you evaluate open source models on the Game24 task and prepare a submission for Kaggle competitions.

## Overview

The Game24 task involves finding mathematical expressions using four given numbers that equal 24. This project uses the TS_LLM framework with pre-trained LLaMA-2 7B models fine-tuned specifically for Game24.

## Available Models

- **Policy Model**: `OhCherryFire/llama2-7b-game24-policy-hf` - Generates step-by-step solutions
- **Value Model**: `OhCherryFire/llama2-7b-game24-value` - Evaluates solution quality

## Quick Start

### 1. Environment Setup

First, make the setup script executable and run it:

```bash
chmod +x setup_game24_evaluation.sh
./setup_game24_evaluation.sh
```

This will:
- Create a conda environment named `tsllm`
- Install all required dependencies
- Download the pre-trained Game24 models
- Convert the policy model to CTranslate2 format for faster inference

### 2. Run Evaluation

Make the evaluation script executable and run it:

```bash
chmod +x evaluate_game24.sh
./evaluate_game24.sh
```

This will evaluate the models using three different methods:
- **MCTS (Monte Carlo Tree Search)**: Uses tree search with value guidance
- **Chain-of-Thought Self-Consistency**: Generates multiple solutions and votes
- **Chain-of-Thought Greedy**: Single greedy generation

### 3. Format for Kaggle Submission

```bash
python format_kaggle_submission.py --results_dir results --output kaggle_submission/game24_submission.csv
```

## Detailed Instructions

### System Requirements

- **OS**: Linux or macOS (Windows with WSL2)
- **GPU**: NVIDIA GPU with at least 8GB VRAM (recommended)
- **RAM**: At least 16GB system RAM
- **Storage**: At least 20GB free space for models and results

### Manual Installation (Alternative)

If the automated setup fails, you can install manually:

```bash
# Create conda environment
conda create -n tsllm python=3.10
conda activate tsllm

# Install requirements
pip install ctranslate2==3.17.1
pip install transformers==4.29.2
pip install torch==2.0.1+cu117
pip install numpy==1.25.0
pip install flash-attn==1.0.9
pip install dm-tree==0.1.8
pip install huggingface_hub pandas kaggle

# Install the package
pip install -e .

# Download models manually
mkdir -p models/policy models/value models/ct2_cache
cd models/policy
git clone https://huggingface.co/OhCherryFire/llama2-7b-game24-policy-hf
cd ../value
git clone https://huggingface.co/OhCherryFire/llama2-7b-game24-value
cd ../..

# Convert to CTranslate2
ct2-transformers-converter \
    --model models/policy/llama2-7b-game24-policy-hf \
    --quantization bfloat16 \
    --output_dir models/ct2_cache/llama2-7b-game24-policy-ct2
```

### Understanding the Evaluation Methods

#### 1. MCTS (Best Performance)
- Uses Monte Carlo Tree Search with neural network guidance
- Explores multiple solution paths intelligently
- Balances exploration vs exploitation using the value model
- Typically achieves the highest accuracy

#### 2. Chain-of-Thought Self-Consistency
- Generates multiple solution attempts
- Uses majority voting to select the final answer
- Good for problems with multiple valid solution paths
- More robust but computationally expensive

#### 3. Chain-of-Thought Greedy
- Single-pass generation
- Fastest method but may miss optimal solutions
- Good baseline for comparison

### Customizing Evaluation Parameters

You can modify the evaluation parameters by editing the `args_list` in `evaluate_game24.sh`:

```python
args_list = [
    {
        "temperature": 1.0,          # Sampling temperature
        "max_length": 4,             # Maximum tree depth
        "max_action": 20,            # Maximum actions per step
        "pb_c_init": 3,              # MCTS exploration constant
        "num_simulations": 5,        # MCTS simulations per step
        "k_maj": 10,                 # Number of samples for self-consistency
        "rollout_method": "mcts.rap", # Search method
        # ... other parameters
    }
]
```

### Troubleshooting

#### CUDA Out of Memory
If you encounter GPU memory issues:
1. Reduce batch size in the evaluation scripts
2. Use CPU-only mode (slower): set `CUDA_VISIBLE_DEVICES=""`
3. Use model quantization: change quantization to `int8` in the CT2 conversion

#### Model Download Issues
If model download fails:
1. Check your internet connection
2. Ensure you have enough disk space
3. Try downloading manually using `git clone` commands above

#### CTranslate2 Conversion Issues
If CT2 conversion fails:
1. Ensure you have the correct transformers version (4.29.2)
2. Try using `float32` quantization instead of `bfloat16`
3. Check that the model files are complete

### Expected Results

The pre-trained models should achieve:
- **MCTS**: ~85-90% accuracy on the Game24 test set
- **CoT-SC**: ~80-85% accuracy
- **CoT-Greedy**: ~70-75% accuracy

### File Structure After Setup

```
LLM_Tree_Search/
├── models/
│   ├── policy/llama2-7b-game24-policy-hf/
│   ├── value/llama2-7b-game24-value/
│   └── ct2_cache/llama2-7b-game24-policy-ct2/
├── results/
│   ├── mcts_results/
│   ├── cot_sc_results/
│   └── cot_greedy_results/
├── kaggle_submission/
│   └── game24_submission.csv
├── setup_game24_evaluation.sh
├── evaluate_game24.sh
└── format_kaggle_submission.py
```

### Kaggle Submission Format

The generated submission file will have the format:
```csv
id,problem,solution
0,"1 1 4 6","(6 - 4) * (1 + 1) = 24"
1,"1 1 11 11","(11 + 1) * (1 + 1) = 24"
...
```

### Advanced Usage

#### Running with Multiple GPUs
```bash
# Modify the torchrun command in evaluate_game24.sh
torchrun --nproc_per_node=4 --master-port 29522 tsllm/offline_rl/test_sft_and_v.py ...
```

#### Custom Test Dataset
To evaluate on a custom dataset, replace the CSV file:
```bash
cp your_dataset.csv tsllm/envs/game24/24.csv
```

#### Batch Processing
For large-scale evaluation, you can split the dataset and run multiple evaluation jobs in parallel.

## Citation

If you use this code or models in your research, please cite:

```bibtex
@article{feng2023alphazero,
  title={Alphazero-like Tree-Search can guide large language model decoding and training},
  author={Feng, Xidong and others},
  journal={arXiv preprint arXiv:2309.17179},
  year={2023}
}
```

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review the original paper and repository
3. Open an issue on the original GitHub repository