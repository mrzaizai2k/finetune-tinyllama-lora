# finetune-tinyllama-lora
finetune tiny llama


## Question 1: Data Analysis and Model Training
- **Data Analysis**:
  - Created `emotion_eda.ipynb` to analyze the dataset.
  - Due to time constraints, unable to train on full data or generate additional data.
  - Removed data to ensure `max_seq_length` and dataset balance.
- **Model Training**:
  - Trained the model using LoRA in `assessment.py`.
  - Training one model for 5 epochs took 20 minutes due to time limitations.
  - With more time, I would:
    - Search for optimal hyperparameters using tools like Weights & Biases (WandB) or other MLOps methods to log experiments and select the best model.
    - Generate additional data using another model to ensure data balance instead of removing data.
    - Add logs and configuration files for better experiment tracking.
- **Model Comparison**:
  - Comparison between base and fine-tuned models is in `test.ipynb`.
  - Used `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T` as the base model.
  - Noted that comparing with this base model is unfair since it is a foundation model not designed to handle queries directly, leading to 0 accuracy for the classification task.
  - The base model is intended for further fine-tuning, not standalone use.
