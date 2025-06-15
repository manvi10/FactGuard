"""
Fake News Detection using DistilBERT
Dataset: Kaggle Fake and Real News Dataset
Model: DistilBERT (Hugging Face Transformers)
"""

# ========== Imports ==========
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import torch
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments
)
from datasets import Dataset
import transformers

# ========== Logging and Device ==========
transformers.logging.set_verbosity_info()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using GPU:", torch.cuda.is_available())

# ========== Load and Prepare Data ==========
true_df = pd.read_csv("/kaggle/input/fake-and-real-news-dataset/True.csv")
fake_df = pd.read_csv("/kaggle/input/fake-and-real-news-dataset/Fake.csv")

true_df["label"] = 1
fake_df["label"] = 0
df = pd.concat([true_df, fake_df])
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df["content"] = df["title"] + " " + df["text"]
df = df[["content", "label"]]

# ========== Train/Test Split ==========
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df["content"].tolist(), df["label"].tolist(), test_size=0.2, random_state=42
)

# ========== Tokenization ==========
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=512)

# ========== Create Dataset ==========
train_dataset = Dataset.from_dict({**train_encodings, "label": train_labels})
test_dataset = Dataset.from_dict({**test_encodings, "label": test_labels})

# ========== Load Model ==========
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2
).to(device)

# ========== Training Arguments ==========
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    weight_decay=0.01,
    evaluation_strategy="steps",
    eval_steps=100,
    logging_dir="./logs",
    logging_strategy="steps",
    logging_steps=50,
    logging_first_step=True,
    load_best_model_at_end=True,
    report_to="none",  # Disable external logging like wandb
    disable_tqdm=False
)

# ========== Evaluation Metric ==========
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {"accuracy": accuracy_score(labels, preds)}

# ========== Trainer ==========
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

# ========== Training ==========
print("\n Starting training...\n")
trainer.train()

# ========== Evaluation ==========
print("\n Final Evaluation:\n")
eval_output = trainer.evaluate()
print(f" Accuracy: {eval_output['eval_accuracy'] * 100:.2f}%")

# ========== Predictions ==========
preds = trainer.predict(test_dataset)
pred_labels = np.argmax(preds.predictions, axis=1)

# ========== Reports ==========
print("\n Classification Report:\n", classification_report(test_labels, pred_labels))
print(" Confusion Matrix:\n", confusion_matrix(test_labels, pred_labels))
