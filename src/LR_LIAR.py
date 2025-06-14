#Logistic Regression on LIAR Dataset

#importing required libraries
import pandas as pd
import numpy as np
import string
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load train, test, validation sets
columns = ['id', 'label', 'statement', 'subject', 'speaker', 'speaker_job', 
           'state', 'party', 'barely_true', 'false', 'half_true', 'mostly_true', 
           'pants_on_fire', 'context']

train_df = pd.read_csv('/liar-dataset/train.tsv', sep='\t', names=columns)
val_df = pd.read_csv('/liar-dataset/valid.tsv', sep='\t', names=columns)
test_df = pd.read_csv('/liar-dataset/test.tsv', sep='\t', names=columns)

# Combine datasets
df = pd.concat([train_df, val_df, test_df], ignore_index=True)

# Filter only 'true' and 'false' for binary classification (simplifying)
df = df[df['label'].isin(['true', 'false'])]

# Map labels to binary: true = 0, false = 1
df['label'] = df['label'].map({'true': 0, 'false': 1})

# Preprocess text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

df['statement'] = df['statement'].apply(clean_text)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(df['statement'], df['label'], test_size=0.2, random_state=42)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train Logistic Regression
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)
y_pred = model.predict(X_test_tfidf)

# Evaluation
accuracy = accuracy_score(y_test, y_pred) * 100  # in percentage
print(f"Accuracy: {accuracy:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - LIAR Dataset")
plt.show()


