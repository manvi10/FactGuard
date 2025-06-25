"""
Fake News Detection using RoBERTa with Metadata Fusion

Dataset: LIAR Dataset (Kaggle)
Model: Hybrid Model combining RoBERTa (Hugging Face Transformers) and structured metadata features

Features:
- Textual input processed via RoBERTa
- Structured metadata (party, state, truth count history) encoded and fused with language model output
- Data augmentation using synonym replacement via WordNet
- Custom Focal Loss to address class imbalance
"""


# Imports and Setup
import os
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizerFast, RobertaModel, get_scheduler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from nltk.corpus import wordnet
from itertools import chain
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Disable TensorFlow logging if used
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ======= Data Loading & Preprocessing =======

def load_and_process_data(filepath):
    df = pd.read_csv(filepath, sep='\t', header=None)
    df.columns = ['id', 'label', 'statement', 'subject', 'speaker', 'job_title', 'state', 'party',
                  'barely_true_counts', 'false_counts', 'half_true_counts', 'mostly_true_counts',
                  'pants_on_fire_counts', 'context']
    df = df[df['label'].isin(['half-true', 'mostly-true', 'true', 'false', 'barely-true', 'pants-fire'])]
    df['label'] = df['label'].apply(lambda x: 1 if x in ['half-true', 'mostly-true', 'true'] else 0)
    df = df.dropna(subset=['statement'])
    df = df[~((df['label'] == 1) & (df['statement'].str.contains('partly|somewhat|kind of', case=False, na=False)))]
    df['text'] = df['statement'].astype(str) + ' [SEP] ' + df['context'].astype(str)

    # Meta feature encoding
    meta_cols = ['party', 'state', 'barely_true_counts', 'false_counts', 'half_true_counts',
                 'mostly_true_counts', 'pants_on_fire_counts']
    for col in meta_cols[:2]:
        df[col] = df[col].fillna('unknown')
        df[col + '_enc'] = LabelEncoder().fit_transform(df[col])
    for col in meta_cols[2:]:
        df[col] = df[col].fillna(0)
        df[col + '_norm'] = (df[col] - df[col].mean()) / df[col].std()
    return df, meta_cols

def synonym_augment(text):
    words = text.split()
    new_words = []
    for word in words:
        synonyms = wordnet.synsets(word)
        lemmas = set(chain.from_iterable([syn.lemma_names() for syn in synonyms]))
        if lemmas and random.random() < 0.2:
            new_words.append(random.choice(list(lemmas)))
        else:
            new_words.append(word)
    return ' '.join(new_words)

# ======= Dataset Class =======

class LiarDataset(Dataset):
    def __init__(self, df):
        self.texts = df['text'].tolist()
        self.labels = df['label'].tolist()
        self.metadata = df[['party_enc', 'state_enc', 'barely_true_counts_norm', 'false_counts_norm',
                            'half_true_counts_norm', 'mostly_true_counts_norm', 'pants_on_fire_counts_norm']].values

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = tokenizer(self.texts[idx], truncation=True, padding='max_length', max_length=256, return_tensors='pt')
        item = {key: val.squeeze(0) for key, val in enc.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        meta = self.metadata[idx]
        item['meta'] = torch.tensor([int(meta[0]), int(meta[1]), *meta[2:]], dtype=torch.float)
        return item

# ======= Model Definition =======

class HybridModel(nn.Module):
    def __init__(self, meta_dim, num_meta_vals, dropout=0.2):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.dropout = nn.Dropout(dropout)
        self.embeddings = nn.ModuleList([nn.Embedding(n, 16) for n in num_meta_vals[:2]])
        self.num_meta_proj = nn.Linear(5, 16)
        self.meta_proj = nn.Linear(16 * 2 + 16, 64)
        self.classifier = nn.Sequential(
            nn.Linear(self.roberta.config.hidden_size + 64, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2)
        )

    def forward(self, input_ids, attention_mask, meta):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = self.dropout(outputs.last_hidden_state[:, 0, :])
        meta_embeds = torch.cat([emb(meta[:, i].long()) for i, emb in enumerate(self.embeddings)], dim=1)
        meta_num = self.num_meta_proj(meta[:, 2:])
        meta_vec = self.meta_proj(torch.cat([meta_embeds, meta_num], dim=1))
        x = torch.cat((cls_output, meta_vec), dim=1)
        return self.classifier(x)

# ======= Loss Function =======

class FocalLoss(nn.Module):
    def __init__(self, alpha=1.5, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        CE_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-CE_loss)
        loss = self.alpha * (1 - pt) ** self.gamma * CE_loss
        return loss.mean()

# ======= Training & Evaluation =======

def train(model, train_loader, val_loader, optimizer, scheduler, loss_fn, device, epochs=10):
    best_f1 = 0
    patience = 5
    no_improve = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            meta = batch['meta'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, meta)
            loss = loss_fn(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        print(f'Epoch {epoch+1} Train loss: {total_loss / len(train_loader):.4f}')
        val_f1 = evaluate(model, val_loader, device, tag='Validation')
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), 'best_model.pt')
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            print('Early stopping triggered')
            break
    model.load_state_dict(torch.load('best_model.pt'))

def evaluate(model, loader, device, tag='Test'):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            meta = batch['meta'].to(device)
            outputs = model(input_ids, attention_mask, meta)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    print(f'{tag} Accuracy: {acc*100:.2f}% | F1: {f1:.4f}')
    if tag == 'Test':
        print(classification_report(all_labels, all_preds))
        sns.heatmap(confusion_matrix(all_labels, all_preds), annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()
    return f1

# ======= Main =======

if __name__ == '__main__':
    FILEPATH = '/kaggle/input/liar-dataset/train.tsv'
    df, meta_cols = load_and_process_data(FILEPATH)

    # Augment & concat
    df_aug = df.copy()
    df_aug['text'] = df_aug['text'].apply(synonym_augment)
    df = pd.concat([df, df_aug])

    # Split
    train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df['label'], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42)

    # Tokenizer
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')

    # Datasets
    train_ds = LiarDataset(train_df)
    val_ds = LiarDataset(val_df)
    test_ds = LiarDataset(test_df)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16)
    test_loader = DataLoader(test_ds, batch_size=16)

    # Model, loss, optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    meta_counts = [len(LabelEncoder().fit(df[col].astype(str)).classes_) for col in meta_cols[:2]]
    model = HybridModel(meta_dim=7, num_meta_vals=meta_counts).to(device)
    loss_fn = FocalLoss(alpha=1.5, gamma=2)

    optimizer = AdamW(model.parameters(), lr=2e-5)
    num_training_steps = len(train_loader) * 10
    scheduler = get_scheduler("cosine", optimizer=optimizer, num_warmup_steps=100, num_training_steps=num_training_steps)

    # Run training and evaluation
    train(model, train_loader, val_loader, optimizer, scheduler, loss_fn, device, epochs=10)
    evaluate(model, test_loader, device, tag='Test')
