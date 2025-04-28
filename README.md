# Finetune-TinyLlama-LoRA

Fine-tune TinyLlama with LoRA using PEFT.

## Setup

```bash
conda create -n tinyllama python=3.10
conda activate tinyllama
pip install -r setup.txt
```

## Data

The dataset used is [dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion), converted to Alpaca format:

```json
[
  {
    "instruction": "Classify the emotion in this tweet:",
    "input": "i feel disheartened or defeated",
    "output": "sadness"
  }
]
```

## Data Analysis and Model Training

### Data Analysis
- Created `emotion_eda.ipynb` for dataset analysis.
- Due to time constraints, could not train on the full dataset or generate additional data.
- Removed data to balance the dataset and ensure compatibility with `max_seq_length`.

### Model Training
- Trained the model using LoRA in `train.py`.
- Training for 5 epochs took 20 minutes due to time limitations.
- With more time, I would:
  - Perform hyperparameter optimization using tools like Weights & Biases (WandB) or other MLOps methods for experiment logging and model selection.
  - Generate additional data with another model to balance the dataset instead of removing data.
  - Add logs and configuration files for improved experiment tracking.

### Model Comparison
- Comparison between base and fine-tuned models is in `compare_performance.ipynb`.
- Used `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T` as the base model.
- Note: Comparing with this base model is unfair as it is a foundation model not designed for direct query handling, resulting in 0 accuracy for the classification task.
- The base model is intended for further fine-tuning, not standalone use.

## Reference
- [Alpaca Fine-Tuning Guide](https://www.mlexpert.io/blog/alpaca-fine-tuning)