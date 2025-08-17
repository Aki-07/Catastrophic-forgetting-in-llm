# Catastrophic Forgetting Research

This repository contains implementations and demonstrations of catastrophic forgetting phenomena in both simple neural networks and large language models (LLMs), along with several mitigation techniques.

## Overview

Catastrophic forgetting occurs when neural networks trained sequentially on different tasks tend to forget previously learned information when learning new tasks. This project provides practical demonstrations and comparisons of various approaches to mitigate this problem.

## Project Structure

```
catastrophic_forgetting/
├── mlp.py                  # Synthetic task experiments with MLPs
├── llm.py                  # LLM experiments with multilingual data
├── requirements.txt        # Python dependencies

```

## Features

### Synthetic Experiments (mlp.py)
- **Task A**: Moon-shaped classification dataset
- **Task B**: Circle-shaped classification dataset
- **Baseline**: Standard sequential training showing catastrophic forgetting
- **Mitigation Methods**:
  - **Replay Buffer**: Maintains a small buffer of previous task examples
  - **Elastic Weight Consolidation (EWC)**: Regularizes important parameters using Fisher information
  - **Parameter Isolation**: Uses task-specific adapter layers

### LLM Experiments (llm.py)
- **Model**: DistilGPT-2 
- **Dataset**: OPUS Books (English-French translation pairs)
- **Phase A**: Fine-tune on French text
- **Phase B**: Fine-tune on English text
- **Methods Tested**:
  - Sequential fine-tuning (baseline - shows forgetting)
  - Experience replay (mixing old and new data)
  - LoRA adapters (parameter isolation)

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd catastrophic_forgetting
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Synthetic MLP Experiments

Run the basic experiment:
```bash
python mlp.py
```

Quick demo with reduced epochs:
```bash
python mlp.py --fast
```

Customize parameters:
```bash
python mlp.py --epochs_a 30 --epochs_b 30 --lr 0.001 --replay-buf 1500
```

**Key Arguments:**
- `--epochs_a`, `--epochs_b`: Training epochs for each task
- `--lr`: Learning rate
- `--replay-buf`: Size of replay buffer
- `--replay-alpha`: Fraction of batch to use for replay
- `--ewc-lambda`: EWC regularization strength
- `--fast`: Quick demo mode
- `--cpu`: Force CPU usage

### LLM Experiments

Run the full LLM experiment:
```bash
python llm.py
```

Fast demo:
```bash
python llm.py --fast
```

Interactive mode (try different model checkpoints):
```bash
python llm.py --fast --interactive
```

**Key Arguments:**
- `--model`: Base model (default: distilgpt2)
- `--epochs_a`, `--epochs_b`: Training epochs for each language
- `--sample_en`, `--sample_fr`: Number of training examples
- `--replay_ratio`: Fraction of old language data to mix in
- `--fast`: Quick demo with reduced data/epochs
- `--interactive`: Start interactive Q&A after training
- `--skip-train`: Skip training, only interactive mode
- `--cpu`: Force CPU usage

### Interactive LLM Demo

In interactive mode, you can:
- Switch between trained models: `/model baseline`, `/model after_A`, etc.
- Adjust generation parameters: `/temp 0.8`, `/len 100`
- Ask questions or provide prompts in different languages
- Exit with `/exit`

## Results

Based on the synthetic experiments, the mitigation techniques show the following effectiveness at retaining Task A knowledge after learning Task B:

| Method | Task A Retention | Task B Performance |
|--------|-----------------|-------------------|
| Baseline (no mitigation) | 63.1% | 89.1% |
| Replay Buffer | **90.1%** | 87.9% |
| EWC | 66.8% | 88.8% |
| Parameter Isolation | **97.2%** | 85.2% |

### Key Findings

1. **Parameter Isolation** (adapters) provides the best retention of previous knowledge
2. **Replay Buffer** offers excellent performance with minimal overhead
3. **EWC** provides modest improvement over baseline but requires Fisher information computation
4. All methods maintain reasonable performance on the new task

## Visualization

The project generates several plots in the `outputs/` directory:
- Training curves for individual tasks
- Comparison of mitigation methods
- Summary bar chart of retention performance

## Technical Details

### MLP Architecture
- Input: 2D features
- Hidden layers: 64 neurons each with ReLU activation
- Output: 2 classes
- Dropout: 0.1 for regularization

### Adapter Architecture
- Bottleneck design: 64 → 16 → 64 dimensions
- Added residually to base model features
- Task-specific heads for classification

### LLM Setup
- Base model: DistilGPT-2
- LoRA configuration: r=8, α=16, dropout=0.05
- Target modules: attention and projection layers
- Block size: 128 tokens

## Requirements

- Python 3.8+
- PyTorch
- Transformers (≥4.41)
- scikit-learn
- matplotlib
- tqdm
- datasets
- accelerate
- peft

## Contributing

Feel free to open issues or submit pull requests to improve the implementations or add new mitigation techniques.


## License

This project is licensed under the MIT License - see the LICENSE file for details.
