# utils.py
import torch
import torch.nn.functional as F
from .model import model, tokenizer, bert_fakereal_model, bert_tokenizer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict(statement, context, party, state, barely_true, false, half_true, mostly_true, pants_on_fire):
    from .utils import get_meta_tensor  # ensure this function returns proper metadata tensor

    text = statement + " [SEP] " + context
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=256
    )

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    meta_tensor = get_meta_tensor(party, state, barely_true, false, half_true, mostly_true, pants_on_fire).to(device)

    with torch.no_grad():
        logits = model(input_ids, attention_mask, meta_tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]

    label = "REAL" if probs[1] > probs[0] else "FAKE"
    return {"prediction": label, "confidence": float(max(probs))}

def predict_fakereal(text):
    encoded = bert_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=256
    )

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    with torch.no_grad():
        logits = bert_fakereal_model(input_ids, attention_mask)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]

    label = "REAL" if probs[1] > probs[0] else "FAKE"
    return {"prediction": label, "confidence": float(max(probs))}

def get_dummy_meta():
    # Example dummy meta (adjust to match your actual format)
    return torch.tensor([[0, 0, 1, 1, 1, 1, 1]], dtype=torch.float32)

# These should match the order and mappings used during model training
PARTY_MAP = {
    'democrat': 0, 'republican': 1, 'none': 2, 'independent': 3, 'libertarian': 4
}
STATE_MAP = {
    'california': 0, 'texas': 1, 'florida': 2, 'new york': 3, 'illinois': 4,
    'unknown': 5  # fallback
}

def get_meta_tensor(party: str, state: str, barely_true: float, false: float, half_true: float, mostly_true: float, pants_on_fire: float):
    # Clean input
    party = party.lower().strip()
    state = state.lower().strip()

    # Map to IDs (fallback to 'none' / 'unknown')
    party_id = PARTY_MAP.get(party, PARTY_MAP['none'])
    state_id = STATE_MAP.get(state, STATE_MAP['unknown'])

    # Combine categorical and numerical inputs into a tensor
    meta_tensor = torch.tensor([[party_id, state_id, barely_true, false, half_true, mostly_true, pants_on_fire]], dtype=torch.float)

    return meta_tensor
