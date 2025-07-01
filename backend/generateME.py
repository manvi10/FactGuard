import pandas as pd
import json
from sklearn.preprocessing import LabelEncoder
import os

# Column names based on LIAR dataset structure
columns = [
    "id", "label", "statement", "subject", "speaker", "job", 
    "state", "party", "barely_true", "false", "half_true", 
    "mostly_true", "pants_on_fire", "context"
]

# Correct path and read with header=None
df = pd.read_csv(
    r"C:\Projects\FactGuard\data\LIAR Dataset\train.tsv",
    sep="\t",
    header=None,
    names=columns,
    quoting=3  # avoid issues if there are quote characters
)

print("Columns loaded:", df.columns)

# Keep only 'party' and 'state'
df = df[["party", "state"]].dropna()

# Encode
encoders = {}
for col in df.columns:
    le = LabelEncoder()
    le.fit(df[col])
    encoders[col] = list(le.classes_)

# Save
os.makedirs("models/BERT_LIAR", exist_ok=True)
with open("models/BERT_LIAR/meta_encoders.json", "w") as f:
    json.dump(encoders, f, indent=2)

print("meta_encoders.json saved successfully!")
