import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType

# LLM Configuration
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct" 
OUTPUT_DIR = "./output/spam_adapter"
MAX_LENGTH = 128  # Keep short for CPU speed

# Format: "Classify: <text> \n Label: <spam/ham>"
def format_prompt(example):
    # Map dataset label (0=ham, 1=spam) to text
    label_text = "spam" if example['label'] == 1 else "ham"
    prompt = f"Classify this SMS as 'spam' or 'ham'.\nSMS: {example['sms']}\nLabel: {label_text}"
    return {"text": prompt}

# Prepare Dataset (Using SMS Spam Collection)
def prepare_data():
    # Loading the SMS Spam Collection downloaded from kaggle
    dataset = load_dataset("csv", data_files="data/spam.csv", split="train", encoding="latin-1")

    if "v1" in dataset.column_names and "v2" in dataset.column_names:
        dataset = dataset.rename_columns({"v1": "label", "v2": "sms"})
    
    dataset = dataset.map(lambda x: {"label": 1 if x["label"] == "spam" else 0})

    dataset = dataset.train_test_split(test_size=0.2)  # Split: 80% train, 20% validation
    
    # Format the data
    dataset = dataset.map(format_prompt)
    return dataset

# Load Tokenizer & Model
def train():
    print("Loading model")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    # Apply PEFT (LoRA)
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,            # Rank
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"] # Target attention layers
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Tokenization
    dataset = prepare_data()
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=MAX_LENGTH)
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # Training Arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4, # Low batch size for CPU
        gradient_accumulation_steps=4, # Simulate larger batch
        num_train_epochs=1,            # 1 epoch is enough for this simple task
        learning_rate=2e-4,
        logging_steps=10,
        save_strategy="epoch",
        use_cpu=True,                  # Use GPU training
        report_to="none"               # Disable wandb
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    print("Starting training...")
    trainer.train()
    
    print("Saving model")
    trainer.save_model(OUTPUT_DIR)
    print("Training Completed")

if __name__ == "__main__":
    train()