import torch
from .model import model, tokenizer, bert_fakereal_model, bert_tokenizer
from .utils import get_dummy_meta, get_meta_tensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Token cleaner ===
def clean_token(token):
    if token.startswith("Ġ"):
        return token[1:]
    elif token in ["[PAD]", "[SEP]", "[CLS]"]:
        return ""
    return token

# ====== Attention Explainer for RoBERTa + Metadata (LIAR) ======
def explain(text: str):
    """Explain LIAR statement using attention weights from RoBERTa"""
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=256,
        return_attention_mask=True,
        return_token_type_ids=False
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    meta_tensor = get_dummy_meta().to(device)

    with torch.no_grad():
        _, attentions = model(input_ids, attention_mask, meta_tensor, return_attentions=True)

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    last_layer_attention = attentions[-1][0][0]  # [seq_len, seq_len]
    cls_attention = last_layer_attention[:, 0].cpu().numpy()

    explanation = [
        (clean_token(tok), float(score))
        for tok, score in zip(tokens, cls_attention)
        if clean_token(tok) != ""
    ]
    explanation.sort(key=lambda x: abs(x[1]), reverse=True)
    return explanation[:10]

# ====== Attention Explainer for DistilBERT Fake/Real Model ======
def explain_fakereal(text: str):
    """Explain Fake/Real statement using attention weights from DistilBERT"""
    encoded = bert_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=256,
        return_attention_mask=True,
        return_token_type_ids=False
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    with torch.no_grad():
        _, attentions = bert_fakereal_model(input_ids, attention_mask, return_attentions=True)

    tokens = bert_tokenizer.convert_ids_to_tokens(input_ids[0])
    last_layer_attention = attentions[-1][0][0]
    cls_attention = last_layer_attention[:, 0].cpu().numpy()

    explanation = [
        (clean_token(tok), float(score))
        for tok, score in zip(tokens, cls_attention)
        if clean_token(tok) != ""
    ]
    explanation.sort(key=lambda x: abs(x[1]), reverse=True)
    return explanation[:10]
