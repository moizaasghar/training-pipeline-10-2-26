import os
import wandb
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from datasets import load_dataset, DatasetDict, concatenate_datasets
import numpy as np
import evaluate
from dotenv import load_dotenv

load_dotenv()

api_key = os.environ.get("WANDB_API_KEY")
testing = os.environ.get("TESTING", "false").lower() 
run_name = os.environ.get("WANDB_RUN_NAME")  
dataset_name = os.environ.get("DATASET_NAME") 

if api_key is None:
    raise EnvironmentError("WANDB_API_KEY not found in environment variables.")

# Log in to Weights & Biases using the API key
wandb.login(key=api_key)

# Initialize a new W&B run for experiment tracking
run = wandb.init(project="bert-tiny-sentiment-10-2-26", name=run_name, job_type="training")

dataset = load_dataset(dataset_name)
dataset = DatasetDict({
    "train": dataset["train"],
    "test": dataset["test"]
})

print(f"Dataset size: {len(dataset['train'])} training samples, {len(dataset['test'])} test samples")

if testing == "true":
    dataset["train"] = dataset["train"].select(range(10))
    dataset["test"] = dataset["test"].select(range(1))
    print(f"FOR TESTING: Reduced dataset size: {len(dataset['train'])} training samples, {len(dataset['test'])} test samples")

tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")

def tokenize_fn(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=256)

encoded = dataset.map(tokenize_fn, batched=True)

encoded.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

model = AutoModelForSequenceClassification.from_pretrained("prajjwal1/bert-tiny", num_labels=2)

accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy.compute(predictions=predictions, references=labels)
    f1_score = f1.compute(predictions=predictions, references=labels, average="weighted")
    return {
        "accuracy": acc["accuracy"],
        "f1": f1_score["f1"]
    }
training_args = TrainingArguments(
    warmup_ratio=0.1,                        # Fraction of steps for learning rate warmup
    lr_scheduler_type="cosine",               # Use cosine learning rate scheduler
    learning_rate=2e-5,                       # Initial learning rate
    max_grad_norm=1.0,                        # Gradient clipping
    save_safetensors=True,                    # Save model in safetensors format
    output_dir="./bert-tiny-sentiment",       # Directory to save model checkpoints
    per_device_train_batch_size=8,            # Batch size for training
    per_device_eval_batch_size=8,             # Batch size for evaluation
    num_train_epochs=2,                       # Number of training epochs
    eval_strategy="steps",                    # Evaluate every N steps
    save_strategy="steps",                    # Save checkpoint every N steps
    save_steps=1000,                          # Save checkpoint every 1000 steps
    eval_steps=100,                          # Evaluate every 1000 steps
    logging_dir="./logs",                     # Directory for logs
    logging_steps=50,                         # Log every 50 steps
    load_best_model_at_end=True,              # Load the best model at the end of training
    metric_for_best_model="f1",               # Use F1 score to select the best model
    greater_is_better=True,                   # Higher F1 is better
    report_to="wandb",                        # Report metrics to W&B
    run_name="bert-tiny-imdb-run",            # Name of the W&B run
    save_total_limit=2,                       # Keep only the 2 most recent checkpoints
)

# Initialize the Trainer with model, data, metrics, and callbacks
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded["train"],
    eval_dataset=encoded["test"],
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]  # Stop early if no improvement for 2 evals
)

trainer.train()

model.config.id2label = {0: "Negative", 1: "Positive"}
model.config.label2id = {"Negative": 0, "Positive": 1}

model.save_pretrained("bert-tiny-imdb")
tokenizer.save_pretrained("bert-tiny-imdb")

# Log the trained model as a W&B artifact
model_artifact = wandb.Artifact(name="bert-tiny-imdb-model", type="model")
model_artifact.add_dir("bert-tiny-imdb")
run.log_artifact(model_artifact)

# Save the train and test splits as CSV files for reproducibility
dataset["train"].to_csv("train_split.csv")
dataset["test"].to_csv("test_split.csv")

# Log the dataset splits as a W&B artifact
data_artifact = wandb.Artifact(name="imdb-sentiment-dataset-splits", type="dataset")
data_artifact.add_file("train_split.csv")
data_artifact.add_file("test_split.csv")
run.log_artifact(data_artifact)

# Finish the W&B run
run.finish()