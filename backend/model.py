import torch
import torch.nn as nn
import json
from pathlib import Path
from transformers import (
    RobertaModel, RobertaTokenizerFast,
    DistilBertModel, DistilBertTokenizerFast
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== Hybrid RoBERTa + Metadata Model (LIAR Dataset) =====
class HybridModel(nn.Module):
    def __init__(self, meta_dim, num_meta_vals, dropout=0.2):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained("roberta-base", output_attentions=True)
        self.dropout = nn.Dropout(dropout)

        # Categorical metadata: party, state
        self.embeddings = nn.ModuleList([nn.Embedding(n, 16) for n in num_meta_vals[:2]])

        # Numerical metadata projection
        self.num_meta_proj = nn.Linear(5, 16)

        # Combine all metadata
        self.meta_proj = nn.Linear(16 * 2 + 16, 64)

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.roberta.config.hidden_size + 64, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2)
        )

    def forward(self, input_ids, attention_mask, meta, return_attentions=False):
        roberta_outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = self.dropout(roberta_outputs.last_hidden_state[:, 0, :])

        meta_embeds = torch.cat([emb(meta[:, i].long()) for i, emb in enumerate(self.embeddings)], dim=1)
        meta_num = self.num_meta_proj(meta[:, 2:])
        meta_vec = self.meta_proj(torch.cat([meta_embeds, meta_num], dim=1))

        combined = torch.cat((cls_output, meta_vec), dim=1)
        logits = self.classifier(combined)

        return (logits, roberta_outputs.attentions) if return_attentions else logits

# ===== Load RoBERTa Hybrid Model =====
MODEL_DIR = Path("models/BERT_LIAR")

with open(MODEL_DIR / "model_config.json") as f:
    config = json.load(f)

tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_DIR)
model = HybridModel(
    meta_dim=config["meta_dim"],
    num_meta_vals=config["num_meta_vals"]
)
model.load_state_dict(torch.load(MODEL_DIR / "model_weights.pt", map_location=device))
model.to(device)
model.eval()

# ===== DistilBERT Base Model (Fake/Real Dataset) =====
class DistilBERTFakeRealModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.distilbert = DistilBertModel.from_pretrained("distilbert-base-uncased", output_attentions=True)
        self.pre_classifier = nn.Linear(self.distilbert.config.hidden_size, self.distilbert.config.hidden_size)
        self.classifier = nn.Linear(self.distilbert.config.hidden_size, 2)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()

    def forward(self, input_ids, attention_mask, return_attentions=False):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = outputs.last_hidden_state  # (bs, seq_len, hidden_size)
        pooled_output = hidden_state[:, 0]        # (bs, hidden_size)
        pooled_output = self.pre_classifier(pooled_output)
        pooled_output = self.relu(pooled_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return (logits, outputs.attentions) if return_attentions else logits

# ===== Load DistilBERT Fake/Real Model =====
FAKEREAL_DIR = Path("models/BERT_FAKEREAL")
bert_tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
bert_fakereal_model = DistilBERTFakeRealModel()
bert_fakereal_model.load_state_dict(torch.load(FAKEREAL_DIR / "bert_fakereal_weights.pt", map_location=device))
bert_fakereal_model.to(device)
bert_fakereal_model.eval()
