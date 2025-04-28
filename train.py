import torch
import json
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

# Configurations
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
OUTPUT_DIR = "./model"
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "v_proj"]
BATCH_SIZE = 128
MICRO_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
LEARNING_RATE = 3e-4
TRAIN_STEPS = 300
WARMUP_STEPS = 100
EVAL_STEPS = 50
SAVE_STEPS = 50
LOGGING_STEPS = 10
MAX_SEQ_LENGTH = 64

# Label mapping
LABEL_MAP = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise"
}

def process_dataset(dataset, split, output_file):
    """Process dataset into instruction format, filter by token length, balance classes, and save as JSON."""
    # Load tokenizer for token length calculation
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    data = dataset[split]
    # Create a list of processed examples with their token lengths
    processed_data = []
    for example in data:
        instruction = "Classify the emotion in this tweet:"
        input_text = example["text"]
        output = LABEL_MAP[example["label"]]
        formatted_text = f"{instruction} {input_text} {output}"
        token_length = len(tokenizer(formatted_text, add_special_tokens=False)["input_ids"])
        if token_length <= MAX_SEQ_LENGTH:
            processed_data.append({
                "instruction": instruction,
                "input": input_text,
                "output": output,
                "token_length": token_length
            })

    # Group by emotion to balance classes
    from collections import defaultdict
    grouped_data = defaultdict(list)
    for example in processed_data:
        grouped_data[example["output"]].append(example)

    # Find the minimum number of examples (given as 572 for surprise)
    min_count = 572

    # Sample min_count examples from each class
    balanced_data = []
    for emotion in LABEL_MAP.values():
        class_data = grouped_data[emotion]
        # Shuffle and sample
        import random
        random.shuffle(class_data)
        balanced_data.extend(class_data[:min_count])

    # Remove token_length key as it's not needed in the final dataset
    final_data = [{"instruction": ex["instruction"], "input": ex["input"], "output": ex["output"]} for ex in balanced_data]

    # Save to JSON
    with open(output_file, 'w') as f:
        json.dump(final_data, f, indent=2)
    return final_data

def load_json_dataset(file_path):
    """Load dataset from JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def tokenize_function(examples, tokenizer, max_length):
    """Tokenize the formatted text."""
    formatted_texts = [
        f"{ex['instruction']} {ex['input']} {ex['output']}"
        for ex in examples
    ]
    tokenized = tokenizer(
        formatted_texts,
        padding=True,
        max_length=max_length,
        return_tensors="pt"
    )
    return {
        "input_ids": tokenized["input_ids"].tolist(),
        "attention_mask": tokenized["attention_mask"].tolist(),
        "label_text": [ex["output"] for ex in examples]
    }

def prepare_model(model_name):
    """Load and prepare the model with LoRA."""
    model = AutoModelForCausalLM.from_pretrained(model_name)
    config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model

def setup_training_args(output_dir):
    """Set up training arguments."""
    return TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=MICRO_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=WARMUP_STEPS,
        max_steps=TRAIN_STEPS,
        learning_rate=LEARNING_RATE,
        fp16=True,
        logging_steps=LOGGING_STEPS,
        evaluation_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=3,
        load_best_model_at_end=True,
        # report_to="tensorboard",
        optim="adamw_torch"
    )

def evaluate_model(model, tokenizer, test_data, token_to_label):
    """Evaluate the model on the test set."""
    model.eval()
    correct = 0
    total = 0
    for example in test_data:
        input_text = f"{example['instruction']} {example['input']}"
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=1,
                pad_token_id=tokenizer.eos_token_id
            )
        generated_token = outputs[0, -1].item()
        predicted_label = token_to_label.get(generated_token, None)
        true_label = example["output"]
        if predicted_label == true_label:
            correct += 1
        total += 1
    accuracy = correct / total if total > 0 else 0
    return accuracy

def main():
    # Load original dataset
    dataset = load_dataset("emotion")

    # Process and save datasets
    train_data = process_dataset(dataset, "train", "train_data.json")
    val_data = process_dataset(dataset, "validation", "val_data.json")
    test_data = process_dataset(dataset, "test", "test_data.json")

    # Load processed datasets
    train_data = load_json_dataset("train_data.json")
    val_data = load_json_dataset("val_data.json")
    test_data = load_json_dataset("test_data.json")

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Tokenize datasets
    tokenized_train = tokenize_function(train_data, tokenizer, MAX_SEQ_LENGTH)
    tokenized_val = tokenize_function(val_data, tokenizer, MAX_SEQ_LENGTH)

    # Convert to Dataset for Trainer
    from datasets import Dataset
    train_dataset = Dataset.from_dict(tokenized_train)
    val_dataset = Dataset.from_dict(tokenized_val)

    # Prepare model
    model = prepare_model(MODEL_NAME)

    # Setup training
    training_args = setup_training_args(OUTPUT_DIR)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator
    )

    # Train
    model.config.use_cache = False
    trainer.train()
    model.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    main()